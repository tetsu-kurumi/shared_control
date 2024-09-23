import sys
import ast
import os
import time
from datetime import datetime, timedelta
import torch
import gc
import speech_recognition as sr
import threading
import numpy as np
import scipy.special
import pandas as pd
from openai import OpenAI

import whisper

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared_control_proj.shared_control import config, llm_prompting

class UtteranceInference:
    def __init__(self):
        # Initialize the recognizer
        # self.recognizer = sr.Recognizer()
        # Initialize LLM
        self.llm = llm_prompting.LLM()
        self.storage = ['']
        self.lock = threading.Lock()
        self.pg = self.init_log_pg()  # Store probabilities in log-space
        self.initial_pg = self.pg.copy()

        self._log = {
            "time_stamp": [],
            "letter_num": [],
            "holding_letter": [],
            "reset": [],
            "utterance": [],
            "inference": [],
            "inference_time": [],
            "pg": [],
            "valid_pick_pos": [],
            "valid_place_pos": [],
        }
        self.valid_pick_pos = np.zeros(len(config.SLOT_POS_LIST))
        self.valid_place_pos = np.zeros(len(config.SLOT_POS_LIST))
        
        # The last time a recording was retrieved from the queue.
        self.phrase_time = None
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = config.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        # Important for linux users.
        # Prevents permanent application hang and crash by using the wrong Microphone
        if 'linux' in platform:
            mic_name = config.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            self.source = sr.Microphone(sample_rate=16000)

        # Load / Download model
        self.model = config.model
        if config.model != "large" and not config.non_english:
            self.model = self.model + ".en"
        self.audio_model = whisper.load_model(self.model)

        self.record_timeout = config.record_timeout
        self.phrase_timeout = config.phrase_timeout

        self.phrase_complete = False

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

        # Cue the user that we're ready to go.
        print("Model loaded.\n")

        self.stop = False

        self.listening_thread = threading.Thread(target=self.recognize_speech_from_mic)
        self.update_pg_thread = threading.Thread(target=self._update)

        print("[STUDY MESSAGE] Initializing the speech recognizer")
        self.listening_thread.start()
        # time.sleep(3)
        self.update_pg_thread.start()

        self.transcribe_timer_start = None
        self.inference_timer_start = None

    def stop_threads(self):
        self.stop = True
        print("Stopping transcription and clearing GPU memory.")
        self.clear_gpu_memory()
        self.listening_thread.join()
        self.update_pg_thread.join()
    
    def record_callback(self, _, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def clear_gpu_memory(self):
        # Clear GPU memory if using PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # General garbage collection
        gc.collect()

    def init_log_pg(self):
        """Initialize the log-probability distribution."""
        goals = config.SLOT_POS_LIST
        n = len(goals)
        equal_prob = np.log(1 / n)  # Log probability for uniform distribution
        log_probabilities = np.full(n, equal_prob)
        
        return log_probabilities
    
    def is_silent(self, audio_data):
        raw_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate the amplitude
        amplitude = np.abs(raw_data).mean()
        
        # Determine if it's silent based on the threshold
        return amplitude < config.AUDIO_THRESHOLD
    
    def is_english(self, utterance):
        return any(char.isalpha() for char in utterance)
        # allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+=-[]{}\"\'.,<>/?|~` \n')
        # return all(char in allowed_chars for char in utterance)

    def utterance_is_valid(self, utterance):
        return self.is_english(utterance) and (utterance not in config.HALLUCINATIONS)

    # Function to recognize and print speech in chunks
    def recognize_speech_from_mic(self):
        while self.stop == False and not config.DONE:
            try:
                now = datetime.now()
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    self.phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    self.phrase_time = now
                    
                    # Combine audio data from queue
                    audio_data = b''.join(self.data_queue.queue)
                    self.data_queue.queue.clear()
                    
                    # Convert in-ram buffer to something the model can use directly without needing a temp file.
                    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Read the transcription.
                    result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if self.phrase_complete:
                        with self.lock:
                            self.storage.append(text)
                    else:
                        with self.lock:
                            self.storage[-1] = text
                else:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                break
        
    def _get_utterance(self):
        # Join all elements of self.storage into a single string, stripping any extra spaces
        utterance = ''.join(self.storage).strip()

        # Clear storage by resetting it to a list with an empty string
        self.storage = ['']
        return utterance

    def _update(self):
        while self.stop == False and not config.DONE:
            try:
                if self.storage != "":
                    utterance = self._get_utterance()
                    if self.utterance_is_valid(utterance):
                        self.inference_timer_start = datetime.now()
                        inference = self.llm.infer_goal(utterance)
                        try:
                            inference_python = ast.literal_eval(inference)
                            self._update_pg(inference_python)
                            inference_time = datetime.now()- self.inference_timer_start
                            self._log["time_stamp"].append(datetime.now())
                            self._log["letter_num"].append(config.CURRENT_LETTER_NUM)
                            self._log["holding_letter"].append(config.HOLDING_LETTER)
                            self._log["reset"].append(config.RESETTING)
                            self._log["utterance"].append(utterance)
                            self._log["inference"].append(inference)
                            self._log["inference_time"].append(inference_time)
                            self._log["pg"].append(self.get_prob())
                            self.valid_pick_pos[config.get_valid_pick_pos_idx()] = 1
                            self.valid_place_pos[config.get_valid_place_pos_idx()] = 1
                            self._log["valid_pick_pos"].append(self.valid_pick_pos)
                            self._log["valid_place_pos"].append(self.valid_place_pos)


                        except:
                            pass
    
                # Wait for other processing to happen
                time.sleep(0.1)

            except KeyboardInterrupt:
                break

    def _update_distribution(self, log_prior, observation, confidence):
        """
        Update the log-probability distribution based on the observation and its confidence level.
        
        :param log_prior: NumPy array representing the prior log-probability distribution.
        :param observation: List of tuples (index, probability) where index is the position in the prior 
                            distribution, and probability is the observed probability of that index.
        :param confidence: Confidence level of the observation, a value between 0 and 1.
        :return: Updated log-probability distribution as a NumPy array.
        """
        
        # Convert observation to log-space
        log_obs_array = np.full_like(log_prior, -np.inf)  # Log-space for zero probabilities
        for index, prob in observation:
            if prob > 0:
                log_obs_array[index] = np.log(prob)
        
        # Calculate the log-likelihood as a weighted version of the observation
        log_confidence = np.log(confidence)
        confidence = np.clip(confidence, 1e-10, 0.95)
        log_likelihood = np.logaddexp(log_confidence + log_obs_array, np.log(1 - confidence) + log_prior)
        
        # Add a small value to avoid 0
        # log_likelihood = config.add_noise_log_prob(log_likelihood)
        log_likelihood = np.where(log_likelihood == -np.inf, config.NOISE, log_likelihood)
        
        # Normalize the log-probabilities using logsumexp
        log_posterior = log_likelihood - scipy.special.logsumexp(log_likelihood)
        
        return log_posterior

    def _update_pg(self, inference):
        if not isinstance(inference, list):
            return
        
        inferred_goals, learning_rate = inference  # Observation is a tuple

        if len(inferred_goals) == 0 or learning_rate == 0:
            return

        if isinstance(inferred_goals, list):
            inferred_goals = inferred_goals[0]
        
        if isinstance(inferred_goals, dict):
            observation = []
            for inferred_goal, probability in inferred_goals.items():
                # When picking, convert inference from letter to spot
                if not config.HOLDING_LETTER:
                    for key, val in config.LETTERS_PLACED.items():
                        if val == inferred_goal:
                            inferred_goal = key
                            break
                    if inferred_goal in config.INVALID_PICK_SLOTS:
                        return
                else:
                    if inferred_goal in config.INVALID_PLACE_SLOTS:
                        return
                observation.append([config.GOAL_INDICES[inferred_goal], probability])

        
        # Update the log-probability distribution with the new posterior
        new_log_pg = config.clip_probability(self._update_distribution(self.pg, observation, learning_rate))
        
        with self.lock:
            self.pg = new_log_pg

    def get_prob(self):
        """Return the probability distribution in regular probability space."""
        return np.exp(self.pg)  # Convert log-probabilities back to probabilities for output
    
    def get_log_prob(self):
        return self.pg
    
    def flush_letter_probability(self, index):
        with self.lock:
            # Zero out the log probability for the given index
            self.pg[index] = -np.inf  # effectively setting the probability to 0

            # Re-normalize the log probabilities
            self.pg = self.pg - scipy.special.logsumexp(self.pg)

            # Clip the probabilities if required
            self.pg = config.clip_probability(self.pg)

    def get_log(self):
        if len(self._log['time_stamp']) == 0:
            return pd.DataFrame()
        else:
            return pd.concat([
            pd.DataFrame(self._log["time_stamp"], columns=["time_stamp"]),
            pd.DataFrame(self._log["letter_num"], columns=["letter_num"]),
            pd.DataFrame(self._log["holding_letter"], columns=["holding_letter"]),
            pd.DataFrame(self._log["reset"], columns=["reset"]),
            pd.DataFrame(self._log["utterance"], columns=["utterance"]),
            pd.DataFrame(self._log["inference"], columns=["inference"]),
            pd.DataFrame(self._log["inference_time"], columns=["inference_time"]),
            pd.DataFrame(self._log["pg"], columns=[f"p_g{i}" for i in range(len(self._log["pg"][0]))]),
            pd.DataFrame(self._log["valid_pick_pos"], columns=[f"pick_{i}" for i in range(len(self._log["valid_pick_pos"][0]))]),
            pd.DataFrame(self._log["valid_place_pos"], columns=[f"place_{i}" for i in range(len(self._log["valid_place_pos"][0]))]),
        ], axis=1)

