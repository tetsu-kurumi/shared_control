import sys
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from shared_control_proj.shared_control import config, maxent_pred, move_to_goal, shared_auto

class AssistanceSystem:
    def __init__(self, policies, sensitivity, gamma, speed, input, utterance, utterance_inference):
        self._policies = policies
        self._predictor = maxent_pred.MaxEntPredictor(self._policies, sensitivity)
        self._policy = shared_auto.ContinuousSharedAutoPolicy(self._policies) #, self._policies[0].action_space)
        self._arbitrator = ScaledArbitrator(gamma)
        self._speed = speed
        self._utterance_inference = utterance_inference

        self.input_flag = input
        self.utterance_flag = utterance

        self._log = {
            "time_stamp": [],
            "letter_num": [],
            "holding_letter": [],
            "inference_type": [],
            "x": [],
            "u": [],
            "g": [],
            "actual_pg": [],
            "input_pg": [],
            "utterance_pg": [],
            "joint_pg": [],
            "ag": [],
            "a": [],
            "a_appl": [],
            "vel": [],
            "valid_pick_pos": [],
            "valid_place_pos": [],
        }
  

        # For logging purposes
        self.time_step = -1
        self.valid_pick_pos = np.zeros(len(config.SLOT_POS_LIST))
        self.valid_place_pos = np.zeros(len(config.SLOT_POS_LIST))

    def check_boundaries(self, x, y, delta_x, delta_y):
        # print("delta_x:", delta_x, "delta_y:", delta_y)
        new_x = x + delta_x
        new_y = y + delta_y
        return (new_x > config.BOUNDARIES['x_min'] or delta_x >= 1) and (new_x < config.BOUNDARIES['x_max'] or delta_x <= -1) and (new_y > config.BOUNDARIES['y_min'] or delta_y >= 1) and (new_y < config.BOUNDARIES['y_max'] or delta_y <= -1)
    
    def limit_velocity(self, vel):
        vx, vy = vel

        # Apply threshold to each velocity
        vx = min(config.MAX_VELOCITY, max(-config.MAX_VELOCITY, vx))
        vy = min(config.MAX_VELOCITY, max(-config.MAX_VELOCITY, vy))

        # Return the limited velocities as a tuple
        return (vx, vy)
    
    def eudlid_dist(self, input):
        return math.sqrt(input[0]**2 + input[1]**2)
    
    def reset_prob(self, letter):
        if self.input_flag:
            # 0 out the probability of the letter that was just picked up or placed, and normalize the probabilities again
            self._predictor.flush_letter_probability(config.SLOT_POS_NAMES.index(letter))
        if self.utterance_flag:
            self._utterance_inference.flush_letter_probability(config.SLOT_POS_NAMES.index(letter))

    def log_sum_exp(self, log_probs):
        """Compute log-sum-exp in a numerically stable way."""
        max_log_prob = np.max(log_probs)
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

    def get_joint_pg(self, input_pg, utterance_pg):
        # Element-wise sum of log-probs to get joint log-probability
        joint_log_prob = input_pg + utterance_pg

        # Normalize the log-probabilities (log-sum-exp trick)
        log_z = self.log_sum_exp(joint_log_prob)
        normalized_log_prob = joint_log_prob - log_z
        normalized_log_prob = config.clip_probability(normalized_log_prob)

        # Return normalized log-probabilities
        return np.exp(normalized_log_prob)

    def get_utterance_prob(self):
        if self._utterance_inference is None:
            raise Exception("speech recognizer not initialized")
        return self._utterance_inference.get_prob()

    def clean_pg(self, prob_policy):
        # Delete invalid slots from option and normalize the pg again
        if config.HOLDING_LETTER:
            invalid_slot_indices = [config.SLOT_POS_NAMES.index(elem) for elem in config.INVALID_PLACE_SLOTS]
        else:
            invalid_slot_indices = [config.SLOT_POS_NAMES.index(elem) for elem in config.INVALID_PICK_SLOTS]
        prob_policy[invalid_slot_indices] = 0

        total_sum = np.sum(prob_policy)
        # Check for division by zero
        if total_sum != 0:
            # Normalize the array by dividing by the sum
            prob_policy = prob_policy / total_sum
        
        return prob_policy

    def get_action(self, x, u):
        self.time_step += 1
        self._log['time_stamp'].append(datetime.now()),
        self._log['letter_num'].append(config.CURRENT_LETTER_NUM)
        self._log['holding_letter'].append(config.HOLDING_LETTER),
        self._log['inference_type'].append(config.INFERENCE_TYPE),
        self._log["x"].append(x)
        self._log["u"].append(u)
        self._log["g"].append([p.goal for p in self._policies])
        self._log["ag"].append(self._policy.get_base_actions(x))

        self.valid_pick_pos[config.get_valid_pick_pos_idx()] = 1
        self.valid_place_pos[config.get_valid_place_pos_idx()] = 1
        self._log["valid_pick_pos"].append(self.valid_pick_pos)
        self._log["valid_place_pos"].append(self.valid_place_pos)

        vel = u

        self._predictor.update(x, u)
        input_pg = self.clean_pg(self._predictor.get_prob())
        log_input_pg = self._predictor.get_log_prob()
        self._log["input_pg"].append(input_pg)
        # Process the probabilities inferred from utterance
        utterance_pg = self.clean_pg(self.get_utterance_prob())
        log_utterance_prob = self._utterance_inference.get_log_prob()
        self._log["utterance_pg"].append(utterance_pg)
        joint_pg = self.clean_pg(self.get_joint_pg(log_input_pg, log_utterance_prob))
        # Calculate the joint probability distribution
        self._log["joint_pg"].append(joint_pg)

        input_pg[input_pg < config.PROB_THRESHOLD] = 0
        joint_pg[joint_pg < config.PROB_THRESHOLD] = 0
        utterance_pg[utterance_pg < config.PROB_THRESHOLD] = 0

        if self.input_flag and self.utterance_flag:
            pg = joint_pg
        elif self.input_flag:
            pg = input_pg
        elif self.utterance_flag:
            pg = utterance_pg
        
        
        print('input_pg:', input_pg)
        print('utterance_pg:', utterance_pg)
        print('joint_pg:', joint_pg)

        if self.input_flag or self.utterance_flag:
            a = self._policy.get_action(x, pg) * config.CONTROLLER_SENSITIVITY
            a_appl = self._arbitrator(u, a)

            vel = self._speed*a_appl

            self._log["a_appl"].append(a_appl)
            self._log["a"].append(a)
            self._log["actual_pg"].append(pg)
        
        else:
            a = None
            a_appl = None
            vel = self._speed*vel
        
        
        # If about to reach boundaries, set velocity to 0 to stop arm
        if not self.check_boundaries(x[0], x[1], vel[0] * config.DELTA_T, vel[1] * config.DELTA_T):
            vel = vel*0
        
        # Check for min and max velocity
        vel = self.limit_velocity(vel)
        
        self._log["vel"].append(vel)

        return vel
    
    def get_ui_log(self):
        return self._utterance_inference.get_log()
    
    def get_log(self):
        if len(self._log['x']) == 0:
            return pd.DataFrame()
        elif self.input_flag or self.utterance_flag:
            return pd.concat([
            pd.DataFrame(self._log["time_stamp"], columns=["time_stamp"]),
            pd.DataFrame(self._log["letter_num"], columns=["letter_num"]),
            pd.DataFrame(self._log["holding_letter"], columns=["holding_letter"]),
            pd.DataFrame(self._log["inference_type"], columns=["inference_type"]),
            pd.DataFrame(self._log["x"], columns=["x", "y"]),
            pd.DataFrame(self._log["u"], columns=["u_x", "u_y"]),
            pd.DataFrame(self._log["a"], columns=["a_x", "a_y"]),
            pd.DataFrame(self._log["a_appl"], columns=["a_appl_x", "a_appl_y"]),
            pd.DataFrame(self._log["vel"], columns=["vel_x", "vel_y"]),
            pd.DataFrame(self._log["actual_pg"], columns=[f"p_g{i}" for i in range(len(self._log["actual_pg"][0]))]),
            pd.DataFrame(self._log["input_pg"], columns=[f"i_p_g{i}" for i in range(len(self._log["input_pg"][0]))]),
            pd.DataFrame(self._log["utterance_pg"], columns=[f"u_p_g{i}" for i in range(len(self._log["utterance_pg"][0]))]),
            pd.DataFrame(self._log["joint_pg"], columns=[f"c_p_g{i}" for i in range(len(self._log["joint_pg"][0]))]),
            pd.DataFrame([
                { f"a_g{i}_{['x','y'][j]}": v for i,g in enumerate(row) for j,v in enumerate(g)}
                for row in self._log["ag"]
            ]),
            pd.DataFrame([
                { f"g{i}_{['x','y'][j]}": v for i,g in enumerate(row) for j,v in enumerate(g)}
                for row in self._log["g"]
            ]),
            pd.DataFrame(self._log["valid_pick_pos"], columns=[f"pick_{i}" for i in range(len(self._log["valid_pick_pos"][0]))]),
            pd.DataFrame(self._log["valid_place_pos"], columns=[f"place_{i}" for i in range(len(self._log["valid_place_pos"][0]))]),
        ], axis=1)

        else:
            return pd.concat([
            pd.DataFrame(self._log["time_stamp"], columns=["time_stamp"]),
            pd.DataFrame(self._log["letter_num"], columns=["letter_num"]),
            pd.DataFrame(self._log["holding_letter"], columns=["holding_letter"]),
            pd.DataFrame(self._log["inference_type"], columns=["inference_type"]),
            pd.DataFrame(self._log["x"], columns=["x", "y"]),
            pd.DataFrame(self._log["u"], columns=["u_x", "u_y"]),
            pd.DataFrame(self._log["vel"], columns=["vel_x", "vel_y"]),
            pd.DataFrame(self._log["input_pg"], columns=[f"i_p_g{i}" for i in range(len(self._log["input_pg"][0]))]),
            pd.DataFrame(self._log["utterance_pg"], columns=[f"u_p_g{i}" for i in range(len(self._log["utterance_pg"][0]))]),
            pd.DataFrame(self._log["joint_pg"], columns=[f"c_p_g{i}" for i in range(len(self._log["joint_pg"][0]))]),
            pd.DataFrame([
                { f"a_g{i}_{['x','y'][j]}": v for i,g in enumerate(row) for j,v in enumerate(g)}
                for row in self._log["ag"]
            ]),
            pd.DataFrame([
                { f"g{i}_{['x','y'][j]}": v for i,g in enumerate(row) for j,v in enumerate(g)}
                for row in self._log["g"]
            ]),
            pd.DataFrame(self._log["valid_pick_pos"], columns=[f"pick_{i}" for i in range(len(self._log["valid_pick_pos"][0]))]),
            pd.DataFrame(self._log["valid_place_pos"], columns=[f"place_{i}" for i in range(len(self._log["valid_place_pos"][0]))]),
        ], axis=1)
        
def get_assistance(input = False, utterance = False, ui = None):
    # build goals
    policies = [
        move_to_goal.MoveToGoalPolicy(g, config.DIST_THRESH, flat_cost=1)
        for g in config.SLOT_POS_LIST
    ]

    return AssistanceSystem(policies, config.SENSITIVITY, config.GAMMA, config.SPEED_POLICY, input, utterance, ui)

class ScaledArbitrator:
    def __init__(self, gamma=0.5):
        self._gamma = gamma
        if gamma > 1 or gamma < 0:
            raise ValueError("Gamma must be between 0 and 1")

    def __call__(self, u_r, u_h):

        return (1-self._gamma)*np.array(u_r) + self._gamma * np.array(u_h)
