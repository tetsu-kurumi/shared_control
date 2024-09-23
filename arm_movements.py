import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from xarm.wrapper import XArmAPI
from shared_control_proj.shared_control import config, assistance

class ArmMovements():
    def __init__(self) -> None:
        self.arm = XArmAPI(config.IP)
        self.init_teleoperation()
        self.block_carrying = None
        self.pick_btn = 0
        self.place_btn = 0

        self.robot_state_log = {
            "time_stamp": [],
            "letter_num": [],
            "holding_letter": [],
            "mode": [],
            "position": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_eff": [],
            "tcp_speed": [],
            "vel_cmd":[]
        }

        self.letter_data = {
            "letter_num": [],
            "letter": [],
            "action_type": [],
            "pick_spot_name": [],
            "pick_spot": [],
            "place_spot_name": [],
            "place_spot": [],
            "pick_start_time": [],
            "pick_end_time": [],
            "pick_time_delta": [],
            "place_start_time": [],
            "place_end_time": [],
            "place_time_delta": [],
        }

        self.pick_spot = None
        self.pick_end_time = None
        self.place_start_time = None
        self.place_end_time = None

    def init_teleoperation(self):
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(5)
        self.arm.set_state(state=0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(2500)

    def init_set_position(self):
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(2500)

    def check_boundaries(self, x, y, delta_x, delta_y):
        # print("delta_x:", delta_x, "delta_y:", delta_y)
        new_x = x + delta_x
        new_y = y + delta_y
        return (new_x > config.BOUNDARIES['x_min'] or delta_x >= 1) and (new_x < config.BOUNDARIES['x_max'] or delta_x <= -1) and (new_y > config.BOUNDARIES['y_min'] or delta_y >= 1) and (new_y < config.BOUNDARIES['y_max'] or delta_y <= -1)

    def close_gripper(self):
        self.arm.set_gripper_position(270, wait=True)

    def open_gripper(self):
        self.arm.set_gripper_position(400, wait=True)

    def pick_block(self, target_slot, slot_name, assist=None, reset=False):
        if self.block_carrying == None:
            self.init_set_position()

            self.arm.set_position(x = target_slot[0], y = target_slot[1], z = config.Z_PICK_BLOCK, speed=config.SPEED, radius = 0, wait=True)
            self.arm.set_gripper_position(config.GRIPPER_CLOSE, wait=True)
            self.arm.set_position(x = target_slot[0], y = target_slot[1], z = config.Z_INIT, speed=config.SPEED, radius = 0, wait=True)

            self.block_carrying = config.CURRENT_POS_TO_LETTER[target_slot]
            config.INVALID_PICK_SLOTS.append(slot_name)
            config.INVALID_PLACE_SLOTS.remove(slot_name)
            # print('INVALID_PICK_SLOTS:', config.INVALID_PICK_SLOTS)
            if slot_name in config.NUMBERS:
                # print("config.POS_TO_LETTER[target_slot]:", config.POS_TO_LETTER[target_slot])
                config.INVALID_PLACE_SLOTS.remove(self.block_carrying)
                config.LETTERS_PLACED[slot_name] = None
                config.NUM_LETTERS_PLACED -= 1
                # print('INVALID_PLACE_SLOTS:', config.INVALID_PLACE_SLOTS)
            
            config.CURRENT_POS_TO_LETTER[target_slot] = None
            config.HOLDING_LETTER = True
            self.pick_spot = slot_name

            # If picking up the block from the guess slots, decrement the word count
            # Flush the probability of the slot from the distribution
            if not reset:
                assist.reset_prob(slot_name)
                config.INFERENCE_TYPE = "place"    
                
            print('[pick] INVALID_PICK_SLOTS:', config.INVALID_PICK_SLOTS)
            print('[pick] INVALID_PLACE_SLOTS:', config.INVALID_PLACE_SLOTS)
            print('[pick] NUM_LETTERS_PLACED:', config.NUM_LETTERS_PLACED)

            self.init_teleoperation()

    def place_block(self, target_slot, slot_name, assist=None, discard = False, reset=False):
        if self.block_carrying:
            self.init_set_position()

            self.arm.set_position(x = target_slot[0], y = target_slot[1], z = config.Z_RELEASE_BLOCK, speed=config.SPEED, radius = 0, wait=True)
            self.arm.set_gripper_position(config.GRIPPER_OPEN, wait=True)
            self.arm.set_position(x = target_slot[0], y = target_slot[1], z = config.Z_INIT, speed=config.SPEED, radius = 0, wait=True)
            placed_block = self.block_carrying
            if slot_name in config.NUMBERS:
                config.INVALID_PLACE_SLOTS.append(self.block_carrying)
                config.LETTERS_PLACED[slot_name] = self.block_carrying
            
            if not discard:
                config.CURRENT_POS_TO_LETTER[target_slot] = self.block_carrying
                config.INVALID_PICK_SLOTS.remove(slot_name)
            
            config.INVALID_PLACE_SLOTS.append(slot_name)
            
            # Flush the probability of the slot from the distribution
            if not reset:
                assist.reset_prob(slot_name)
            
            self.block_carrying = None
            config.HOLDING_LETTER = False
            config.INFERENCE_TYPE = "pick"

            print('[place] INVALID_PICK_SLOTS:', config.INVALID_PICK_SLOTS)
            print('[place] INVALID_PLACE_SLOTS:', config.INVALID_PLACE_SLOTS)
            print('[place] NUM_LETTERS_PLACED:', config.NUM_LETTERS_PLACED)

            self.init_teleoperation()

            return placed_block
    
    def rest_position(self):
        self.init_set_position()

        self.arm.set_position(config.START_POSITION[0], config.START_POSITION[1], config.START_POSITION[2], config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
        self.arm.set_gripper_position(config.GRIPPER_OPEN, wait=True)

    def pick_and_place_all_blocks(self):
        self.init_set_position()
        for letter, pos in config.SLOT_POS.items():
            print("picking up letter", letter)
            self.arm.set_position(pos[0], pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
            self.arm.set_position(pos[0], pos[1], config.Z_PICK_BLOCK_KEYBOARD, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
            self.arm.set_gripper_position(config.GRIPPER_CLOSE, wait=True)
            self.arm.set_position(pos[0], pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
            print("putting down letter", letter)
            self.arm.set_position(pos[0], pos[1], config.Z_RELEASE_BLOCK_KEYBOARD, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
            self.arm.set_gripper_position(config.GRIPPER_OPEN, wait=True)
            self.arm.set_position(pos[0], pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
        current_position =self.arm.get_position()
        self.arm.set_position(current_position[1][0], current_position[1][1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
    
    def reset_blocks(self, reset_block_list, assist):
        self.init_set_position()

        correct_letters = 0

        for job in reset_block_list:
            slot, letter, pick_pos, place_pos = job
            if letter in config.DEAD_LETTER_LIST:
                place_pos = config.DEAD_LETTER_HOLE_POSITION
                config.DISCARDED_LETTERS.append(letter)
                self.arm.set_position(pick_pos[0], pick_pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
                time.sleep(0.05)
                self.pick_block(pick_pos, slot, reset=True)
                self.init_set_position()
                self.arm.set_position(place_pos[0], place_pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
                time.sleep(0.05)
                self.place_block(place_pos, letter, discard=True, reset=True)
                self.init_set_position()

            else:
                if slot != config.ANSWER_POS[letter]:
                    self.arm.set_position(pick_pos[0], pick_pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
                    time.sleep(0.05)
                    self.pick_block(pick_pos, slot, reset=True)
                    self.init_set_position()
                    self.arm.set_position(place_pos[0], place_pos[1], config.Z_INIT, config.DEFAULT_RPY[0], config.DEFAULT_RPY[1], config.DEFAULT_RPY[2], speed=config.SPEED, radius = 0)
                    time.sleep(0.05)
                    self.place_block(place_pos, letter, reset=True)
                    self.init_set_position()
                
                else:
                    config.INVALID_PICK_SLOTS.append(slot)
                    correct_letters += 1
        
        for slot in config.SLOT_POS_NAMES:
            assist.reset_prob(slot)

        self.init_teleoperation()

        config.START = True

        return correct_letters
    
    def determine_action_type(self, placed_spot):
        if self.pick_spot in config.ALPHABETS:
            if placed_spot in config.NUMBERS:
                return config.ACTION_TYPES[0]
            else:
                return config.ACTION_TYPES[3]
        if self.pick_spot in config.NUMBERS:
            if placed_spot in config.ALPHABETS:
                return config.ACTION_TYPES[1]
            elif placed_spot in config.NUMBERS:
                return config.ACTION_TYPES[2]
            else:
                return config.ACTION_TYPES[3]

    def add_letter_data_to_log(self, placed_spot, placed_block):
        self.letter_data["letter_num"].append(config.CURRENT_LETTER_NUM)
        self.letter_data["letter"].append(placed_block)
        self.letter_data["action_type"].append(self.determine_action_type(placed_spot))
        self.letter_data["pick_spot_name"].append(self.pick_spot)
        self.letter_data["pick_spot"].append(config.SLOT_POS[self.pick_spot])
        self.letter_data["place_spot_name"].append(placed_spot)
        self.letter_data["place_spot"].append(config.SLOT_POS[placed_spot])
        self.letter_data["pick_start_time"].append(config.PICK_START_TIME)
        self.letter_data["pick_end_time"].append(self.pick_end_time)
        self.letter_data["pick_time_delta"].append(self.pick_end_time - config.PICK_START_TIME)
        self.letter_data["place_start_time"].append(self.place_start_time)
        self.letter_data["place_end_time"].append(self.place_end_time)
        self.letter_data["place_time_delta"].append(self.place_end_time - self.place_start_time)

    def teleoperation(self, controller, assistance_cond, assist):
        axis_x = controller.get_axis(4)  # Horizontal (X direction)
        axis_y = controller.get_axis(3)  # Vertical (Y direction)
        self.pick_btn = controller.get_button(5)
        self.place_btn = controller.get_button(4)

        current_position = self.arm.get_position()
        current_xy = (current_position[1][0], current_position[1][1])

        if self.pick_btn: 
            # Check the closest block and pick that one up. If there is no object close by, do not go pick up.
            # Check if holding letter
            if self.block_carrying == None:
                # Calculate the euclidean distance and get the closest slot position
                target_slot = config.find_closest_tuple(config.SLOT_POS_LIST, current_xy)
                print('target_slot:', target_slot)
                if target_slot:
                    slot_name = config.POS_TO_LETTER[target_slot]
                    if slot_name:
                        if slot_name not in config.INVALID_PICK_SLOTS:
                            print('slot_name:', slot_name)
                            self.pick_end_time = datetime.now()
                            self.pick_block(target_slot, slot_name, assist=assist)
                            self.place_start_time = datetime.now()
                    
        elif self.place_btn: 
            # Check the closest slot and place it there. If there is no slot close by, do not place
            # Check that we are carrying a block
            if self.block_carrying != None:
                target_slot = config.find_closest_tuple(config.SLOT_POS_LIST, current_xy)
                print('target_slot:', target_slot)
                if target_slot:
                    # Make sure that they can only put it to their own slot or the guess slots
                    slot_name = config.POS_TO_LETTER[target_slot]
                    if slot_name:
                        if slot_name not in config.INVALID_PLACE_SLOTS or slot_name == self.block_carrying:
                            print('slot_name:', slot_name)
                            self.place_end_time = datetime.now()
                            # If not slot, mark the data for this letter as invalid (wrong letter) and set the letter as invalid too. Reset invalid to False when this letter is placed in a slot
                            placed_block = self.place_block(target_slot, slot_name, assist=assist)
                            self.add_letter_data_to_log(slot_name, placed_block)
                            config.CURRENT_LETTER_NUM += 1
                            config.START = True

        # Deadzone to avoid small movements when stick is near center
        deadzone = 0.05

        # Calculate velocity based on joystick input (adjust sensitivity as needed)
        velocity_x = 0
        velocity_y = 0

        # Apply deadzone
        if abs(axis_x) > deadzone:
            velocity_x = axis_x * config.CONTROLLER_SENSITIVITY
            if config.START:
                config.PICK_START_TIME = datetime.now()
                config.START = False
        if abs(axis_y) > deadzone:
            velocity_y = axis_y * config.CONTROLLER_SENSITIVITY
            if config.START:
                config.PICK_START_TIME = datetime.now()
                config.START = False

        if assist is None:
            raise ValueError("assistance required")
        
        u = np.array([velocity_x, velocity_y])
        x = np.array([current_xy[0], current_xy[1]])

        vel = assist.get_action(x, u)
        x_vel = vel[0]
        y_vel = vel[1]

        self.arm.vc_set_cartesian_velocity([x_vel, y_vel, 0, 0, 0, 0])

        self.robot_state_log["time_stamp"].append(datetime.now())
        self.robot_state_log["letter_num"].append(config.CURRENT_LETTER_NUM)
        self.robot_state_log["holding_letter"].append(config.HOLDING_LETTER)
        self.robot_state_log["mode"].append(self.arm.mode)
        self.robot_state_log["position"].append(current_position[1])
        self.robot_state_log["joint_pos"].append(self.arm.get_joint_states()[1][0])
        self.robot_state_log["joint_vel"].append(self.arm.get_joint_states()[1][1])
        self.robot_state_log["joint_eff"].append(self.arm.get_joint_states()[1][2])
        self.robot_state_log["tcp_speed"].append(self.arm.realtime_tcp_speed)
        self.robot_state_log["vel_cmd"].append(vel)
    
    def get_robot_state_log(self):
        if len(self.robot_state_log['time_stamp']) == 0:
            return pd.DataFrame()
        else:
            return pd.concat([
            pd.DataFrame(self.robot_state_log["time_stamp"], columns=["time_stamp"]),
            pd.DataFrame(self.robot_state_log["letter_num"], columns=["letter_num"]),
            pd.DataFrame(self.robot_state_log["holding_letter"], columns=["holding_letter"]),
            pd.DataFrame(self.robot_state_log["mode"], columns=["mode"]),
            pd.DataFrame(self.robot_state_log["position"], columns=["x", "y", "z", "r", "p", "y"]),
            pd.DataFrame(self.robot_state_log["tcp_speed"], columns=["tcp_speed"]),
            pd.DataFrame(self.robot_state_log["joint_pos"], columns=["joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5", "joint_pos_6", "joint_pos_7"]),
            pd.DataFrame(self.robot_state_log["joint_vel"], columns=["joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5", "joint_vel_6", "joint_vel_7"]),
            pd.DataFrame(self.robot_state_log["joint_eff"], columns=["joint_eff_1", "joint_eff_2", "joint_eff_3", "joint_eff_4", "joint_eff_5", "joint_eff_6", "joint_eff_7"]),
            pd.DataFrame(self.robot_state_log["vel_cmd"], columns=["vel_cmd_x", "vel_cmd_y"]),
        ], axis=1)

    def get_letter_data_log(self):
        if len(self.letter_data['letter_num']) == 0:
            return pd.DataFrame()
        else:
            return pd.concat([
            pd.DataFrame(self.letter_data["letter_num"], columns=["letter_num"]),
            pd.DataFrame(self.letter_data["letter"], columns=["letter"]),
            pd.DataFrame(self.letter_data["action_type"], columns=["action_type"]),
            pd.DataFrame(self.letter_data["pick_spot_name"], columns=["pick_spot_name"]),
            pd.DataFrame(self.letter_data["pick_spot"], columns=["pick_spot_x", "pick_spot_y"]),
            pd.DataFrame(self.letter_data["place_spot_name"], columns=["place_spot_name"]),
            pd.DataFrame(self.letter_data["place_spot"], columns=["place_spot_x", "place_spot_y"]),
            pd.DataFrame(self.letter_data["pick_start_time"], columns=["pick_start_time"]),
            pd.DataFrame(self.letter_data["pick_end_time"], columns=["pick_end_time"]),
            pd.DataFrame(self.letter_data["pick_time_delta"], columns=["pick_time_delta"]),
            pd.DataFrame(self.letter_data["place_start_time"], columns=["place_start_time"]),
            pd.DataFrame(self.letter_data["place_end_time"], columns=["place_end_time"]),
            pd.DataFrame(self.letter_data["place_time_delta"], columns=["place_time_delta"]),
        ], axis=1)
        