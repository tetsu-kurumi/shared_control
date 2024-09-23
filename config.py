import math
import numpy as np
from datetime import datetime


def switch_keys_values(input_dict):
    # Using dictionary comprehension to switch keys and values
    switched_dict = {value: key for key, value in input_dict.items()}
    return switched_dict

def euclidean_distance(t1, t2):
    # Calculate the Euclidean distance between two tuples
    return math.sqrt((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2)

def find_closest_tuple(unsorted_tuples, target_tuple):
    # Find the tuple with the minimum Euclidean distance
    closest_tuple = min(unsorted_tuples, key=lambda t: euclidean_distance(t, target_tuple))
    
    # Calculate the distance to the closest tuple
    min_distance = euclidean_distance(closest_tuple, target_tuple)
    
    # Return None if the distance exceeds the threshold, otherwise return the closest tuple
    if min_distance > PICK_PUT_ASSISTANCE_THRESHOLD:
        return None
    else:
        return closest_tuple

ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
NUMBERS = ['1', '2', '3', '4', '5']
INVALID_PICK_SLOTS = NUMBERS.copy()
INVALID_PLACE_SLOTS = ALPHABETS.copy()
DISCARDED_LETTERS = []

# CALIBRATION BLOCKS
# The right edge of the slot
GUESS_X_POS = 296.5
GUESS_Y_REF_POS = -201.2 
# Reference point for y axis of Keyboard first row is Q
KEYBOARD_X_ROW_1 = 240.3
KEYBOARD_Y_REF_POS_ROW_1 = -86.1

KEYBOARD_X_ROW_2 = KEYBOARD_X_ROW_1 + 58.0
KEYBOARD_X_ROW_3 = KEYBOARD_X_ROW_1 + 114.7
# Reference point for y axis of Keyboard second row is A
KEYBOARD_Y_REF_POS_ROW_2 = KEYBOARD_Y_REF_POS_ROW_1 + 25.4
# Reference point for y axis of Keyboard third row is Z
KEYBOARD_Y_REF_POS_ROW_3 = KEYBOARD_Y_REF_POS_ROW_1 + 82.9

SLOT_POS = {
    'A': (KEYBOARD_X_ROW_2 - 1.0, KEYBOARD_Y_REF_POS_ROW_2),
    'B': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3 + 230.5),
    'C': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3 + 115.0),
    'D': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 115.0),
    'E': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 117.0),
    'F': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 173.0),
    'G': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 230.0),
    'H': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 287.5),
    'I': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 401.5),
    'J': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 344.5),
    'K': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 402.0),
    'L': (KEYBOARD_X_ROW_2, KEYBOARD_Y_REF_POS_ROW_2 + 459.0),
    'M': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3 + 344.5),
    'N': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3 + 287.5),
    'O': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 459.5),
    'P': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 516.5),
    'Q': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1),
    'R': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 173.0),
    'S': (KEYBOARD_X_ROW_2 - 1.0, KEYBOARD_Y_REF_POS_ROW_2 + 58.0),
    'T': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 231.0),
    'U': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 344.3),
    'V': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3 + 173.0),
    'W': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 58.0),
    'X': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3 + 57.5),
    'Y': (KEYBOARD_X_ROW_1, KEYBOARD_Y_REF_POS_ROW_1 + 288.0),
    'Z': (KEYBOARD_X_ROW_3, KEYBOARD_Y_REF_POS_ROW_3),
    '1': (GUESS_X_POS, GUESS_Y_REF_POS - 230.0),
    '2': (GUESS_X_POS + 0.5, GUESS_Y_REF_POS - 172.0),
    '3': (GUESS_X_POS, GUESS_Y_REF_POS - 114.5),
    '4': (GUESS_X_POS, GUESS_Y_REF_POS - 57.5),
    '5': (GUESS_X_POS, GUESS_Y_REF_POS)
}

SLOT_POS_LIST = list(SLOT_POS.values())
SLOT_POS_ARR = np.array(SLOT_POS_LIST)
SLOT_POS_NAMES = list(SLOT_POS.keys())
POS_TO_LETTER = switch_keys_values(SLOT_POS)
GOAL_INDICES = {goal: idx for idx, goal in enumerate(SLOT_POS_NAMES)}


def get_avg_slots():
    avg_pos = np.mean(np.array([SLOT_POS_LIST[i] for i in get_valid_pick_pos_idx()]), axis=0)
    return avg_pos

def get_valid_pick_pos_idx():
    valid_slots = [element for element in SLOT_POS_NAMES if element not in INVALID_PICK_SLOTS]
    return [SLOT_POS_NAMES.index(elem) for elem in valid_slots]

def get_valid_place_pos_idx():
    valid_slots = [element for element in SLOT_POS_NAMES if element not in INVALID_PLACE_SLOTS]
    return [SLOT_POS_NAMES.index(elem) for elem in valid_slots]
    

IP = '192.168.1.216'
SPEED = 200
MAX_VELOCITY = 200
MIN_VELOCITY = 10
CONTROLLER_SENSITIVITY = 1
PICK_PUT_ASSISTANCE_THRESHOLD = 20.0
STOP_THRESHOLD = 1.0

DEFAULT_RPY = (180.0, 0.0, 0.0)
Z_INIT = 275.0
Z_RELEASE_BLOCK = 185.0
Z_PICK_BLOCK = 182.0
GRIPPER_OPEN = 400
GRIPPER_CLOSE = 260
START_POSITION = (get_avg_slots()[0], get_avg_slots()[1], Z_INIT)
DEAD_LETTER_HOLE_POSITION = [459.4, -88.9]


BOUNDARIES = {
    'x_min': 200,
    'x_max': 450,
    'y_min': -480,
    'y_max': 467,
}


DELTA_T = 0.05

# NEW VARIABLES
DIST_THRESH = 30 # Half of the distance between the blocks
GAMMA = 0.25
SENSITIVITY = 1
SPEED_POLICY = 200
PROB_THRESHOLD = 0.1
PID_SPEED = 500
NOISE_SCALE = 0.001
NOISE = 1e-12

def add_noise_log_prob(log_probs):
    # Convert log probabilities to linear space
    probs = np.exp(log_probs)
    
    # Add small random noise
    noise = np.random.normal(0, NOISE_SCALE, size=probs.shape)
    noisy_probs = probs + noise
    
    # Ensure probabilities are non-negative and normalize
    noisy_probs = np.clip(noisy_probs, 1e-12, None)  # Avoid negative probabilities
    noisy_probs /= np.sum(noisy_probs)  # Normalize in linear space
    
    # Convert back to log space
    noisy_log_probs = np.log(noisy_probs)
    
    return noisy_log_probs

MAX_PROB_ANY_GOAL = 0.95
LOG_MAX_PROB_ANY_GOAL = np.log(MAX_PROB_ANY_GOAL)
def clip_probability(log_goal_distribution):
    if len(log_goal_distribution) <= 1:
        return log_goal_distribution
    max_prob_ind = np.argmax(log_goal_distribution)
    if log_goal_distribution[max_prob_ind] > LOG_MAX_PROB_ANY_GOAL:
        diff = np.exp(
            log_goal_distribution[max_prob_ind]) - MAX_PROB_ANY_GOAL
        diff_per = diff/(len(log_goal_distribution)-1.)
        
        # NEW
        # Use a stable method for log addition
        other_indices = [i for i in range(len(log_goal_distribution)) if i != max_prob_ind]
        for i in other_indices:
            log_goal_distribution[i] = np.logaddexp(log_goal_distribution[i], np.log(diff_per) - np.log(len(other_indices)))

        log_goal_distribution[max_prob_ind] = LOG_MAX_PROB_ANY_GOAL

    return log_goal_distribution

# LLM variables

SYSTEM_MESSAGE_PLACE = f"""\n[System]: A person is playing Wordle using a robot arm, picking and placing blocks of letter. They are instructed to verbalize their intentions, strategies and thought process to the robot.
You are an agent that is reasoning and inferring which slot the person is intending to control the robot arm to, given the human utterance.
The person is currently carrying a letter block, and is trying to place it to one of the following slots: {[element for element in SLOT_POS_NAMES if element not in INVALID_PLACE_SLOTS]}
Given a human utterance, infer the possible slots that the person is intending to place the letter, and provide a confidence level of your response.

YOU MUST FORMAT YOUR RESPONSE AS A LIST OF A PYTHON DICTIONARY, AND A FLOAT NUMBER. 
The dictionary contains keys WHICH MUST BE ONE OF {[element for element in SLOT_POS_NAMES if element not in INVALID_PLACE_SLOTS]} and values which correspond to the probability that the human's intended goal is the slot in the key per your inference, and the float number is the confidence level of your inference.
The numbers must be in the range of 0 to 1.0, you must choose the goals from the list provided
If there are not enough information to make an inference, you must return an empty list.
Here are example responses you can provide. Example1: [{{'1': 0.25, '2':0.4, '4': 0.3}}, 0.2] Example2: [{{'5': 0.03, '1': 0.2}}, 0.8] Example3: [{{'2': 0.7}}, 0.5] Example4: [{{}}, 0.0].
You must not return anything but the list. 
DO NOT RETURN STRINGS OR EXPLANATIONS.\n"""

SYSTEM_MESSAGE_PICK = f"""\n[System]: A person is playing Wordle using a robot arm, picking and placing blocks of letter. They are instructed to verbalize their intentions, strategies and thought process to the robot.
You are an agent that is reasoning and inferring which slot the person is intending to control the robot arm to, given the human utterance.
The person is currently moving the robot to pick one of the following letters: {[element for element in ALPHABETS if element not in DISCARDED_LETTERS]}
Given a human utterance, infer the possible letter that the person is intending to pick, and provide a confidence level of your response.

YOU MUST FORMAT YOUR RESPONSE AS A LIST OF A PYTHON DICTIONARY, AND A FLOAT NUMBER. 
The dictionary contains keys WHICH MUST BE ONE OF {[element for element in ALPHABETS if element not in DISCARDED_LETTERS]} and values which correspond to the probability that the human's intended goal is the slot in the key per your inference, and the float number is the confidence level of your inference.
The numbers must be in the range of 0 to 1.0, you must choose the goals from the list provided
If there are not enough information to make an inference, you must return an empty list.
Here are example responses you can provide. Example1: [{{'A': 0.25, 'J':0.4}}, 0.2] Example2: [{{'T': 0.03, 'I':0.2, 'G':0.5, 'E': 0.02, 'R': 0.2}}, 0.8] Example3: [{{'H': 0.7}}, 0.5] Example4: [{{}}, 0.0].
You must not return anything but the list. 
DO NOT RETURN STRINGS OR EXPLANATIONS.\n"""
USER_MESSAGE = "[User]: Provide your response for the following utterance: "
DECAY_FACTOR = 0.1
AUDIO_THRESHOLD = 100
LEARNING_RATE = 1

# Speech to Text variables
energy_threshold = 1500
default_microphone = 'pulse'
model = "medium"
non_english = False
record_timeout = 2
phrase_timeout = 3

HALLUCINATIONS = ['you', 'You', 'Thank you.', 'Thank you', 'Thanks for watching!']

# Debugging variables
RUN_DURATION = 30

# Study running variables
NUM_LETTERS_PLACED = 0
LETTERS_PLACED = {'1': None, '2': None, '3': None, '4': None, '5': None}
# LETTERS_PLACED = {'1': 'B', '2': 'R', '3': "O", '4': "W", '5': "A"}

COND1_ANSWER = 'BRAVO'
COND2_ANSWER = 'STING'
COND3_ANSWER = 'STING'
COND4_ANSWER = 'STING'

def construct_answer_pos(answer):
    for idx, letter in enumerate(list(answer)):
        ANSWER_POS[letter] = str(idx + 1)

ANSWER_POS = {}
DEAD_LETTER_LIST = []

CURRENT_POS_TO_LETTER = switch_keys_values({key: SLOT_POS[key] for key in ALPHABETS})


def make_dead_letter_list(answer):
    letters_used = set(answer)
    return [letter for letter in ALPHABETS if letter not in letters_used]

# Logging variables
CURRENT_LETTER_NUM = 0
HOLDING_LETTER = False
ROBOT_STATE_LOG_FILE = None
INFERENCE_LOG_FILE = None
UTTERANCE_LOG_FILE = None
LETTER_DATA_LOG_FILE = None
# META_DATA_LOG_FILE = None
TRAINING = False

ACTION_TYPES = ['keyboard to guess', 'guess to keyboard', 'guess to guess', 'invalid']
START = True
DONE = False
PICK_START_TIME = None

RESETTING = False
INFERENCE_TYPE = "pick"
"""
for participants 0~2
cond2: 1
cond3: 1
cond4: 1
cond12: 1
cond21: 0
cond13: 0
cond31: 1
cond14: 1
cond41: 0

for participants 3~47
cond2: 15
cond3: 15
cond4: 15
cond12: 8
cond21: 7
cond13: 7
cond31: 8
cond14: 8
cond41: 7
"""
PARTCIPANT_COND_DICT = {0: ['cond1', 'cond4'], 1: ['cond3', 'cond1'], 2: ['cond1', 'cond2'], 3: ['cond3', 'cond1'], 4: ['cond1', 'cond4'], 5: ['cond2', 'cond1'], 6: ['cond4', 'cond1'], 7: ['cond1', 'cond3'], 8: ['cond1', 'cond2'], 9: ['cond1', 'cond4'], 10: ['cond2', 'cond1'], 11: ['cond1', 'cond3'], 12: ['cond4', 'cond1'], 13: ['cond1', 'cond4'], 14: ['cond3', 'cond1'], 15: ['cond1', 'cond3'], 16: ['cond1', 'cond4'], 17: ['cond1', 'cond2'], 18: ['cond2', 'cond1'], 19: ['cond2', 'cond1'], 20: ['cond4', 'cond1'], 21: ['cond1', 'cond2'], 22: ['cond4', 'cond1'], 23: ['cond3', 'cond1'], 24: ['cond1', 'cond2'], 25: ['cond3', 'cond1'], 26: ['cond1', 'cond3'], 27: ['cond1', 'cond2'], 28: ['cond1', 'cond4'], 29: ['cond3', 'cond1'], 30: ['cond1', 'cond3'], 31: ['cond3', 'cond1'], 32: ['cond1', 'cond2'], 33: ['cond4', 'cond1'], 34: ['cond1', 'cond3'], 35: ['cond4', 'cond1'], 36: ['cond3', 'cond1'], 37: ['cond3', 'cond1'], 38: ['cond4', 'cond1'], 39: ['cond2', 'cond1'], 40: ['cond1', 'cond2'], 41: ['cond1', 'cond3'], 42: ['cond1', 'cond4'], 43: ['cond2', 'cond1'], 44: ['cond1', 'cond4'], 45: ['cond1', 'cond4'], 46: ['cond1', 'cond2'], 47: ['cond2', 'cond1']}
