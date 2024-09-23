import random
import sys
import os
import time
import pygame
import threading
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared_control_proj.shared_control import config, arm_movements, assistance, utterance_inference


def init_logger(participant_id, cond):
    # Create folder file
    folder_path = create_participant_folder(participant_id)
    
    # Define file paths with participant ID
    config.ROBOT_STATE_LOG_FILE = os.path.join(folder_path, f"robot_state_log_{participant_id}_{cond}.csv")
    config.INFERENCE_LOG_FILE = os.path.join(folder_path, f"inference_log_{participant_id}_{cond}.csv")
    config.UTTERANCE_LOG_FILE = os.path.join(folder_path, f"utterance_log_{participant_id}_{cond}.csv")
    config.LETTER_DATA_LOG_FILE = os.path.join(folder_path, f"letter_data_log_{participant_id}_{cond}.csv")
    # config.META_DATA_LOG_FILE = os.path.join(folder_path, f"meta_data_log_{participant_id}_{cond}.csv")
    
    # Check and create files if they don't exist
    for file_path in [config.ROBOT_STATE_LOG_FILE, config.INFERENCE_LOG_FILE, config.UTTERANCE_LOG_FILE, config.LETTER_DATA_LOG_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("")  # Create an empty file
            print(f"File created: {file_path}")
        else:
            print(f"File already exists: {file_path}")
    
    print("Logger initialized for participant:", participant_id)
    

def create_participant_folder(participant_id, base_directory='./study_data'):
    # Create a new folder path based on the participant ID
    folder_path = os.path.join(base_directory, f"participant_{participant_id}")
    
    # Check if the base directory exists, create it if not
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    
    # Create the folder for the participant if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
    
    return folder_path

def rest_position():
    print("[STUDY MESSAGE] initializing arm")
    # move to start location
    print("[STUDY MESSAGE] moving to start location")
    try:
        arm = arm_movements.ArmMovements()
        arm.rest_position()
        print("[STUDY MESSAGE] initial pose")
    finally:
        None

def init_controller():
    # Initialize Pygame
    code = pygame.init()
    print("[STUDY MESSAGE] pygame initialized with code", code)

    # Initialize Xbox controller
    code = pygame.joystick.init()
    print("[STUDY MESSAGE] joystick initialized with code", code)

    if pygame.joystick.get_count() < 1:
        print("[STUDY MESSAGE] No controller found!")
        exit()

    controller = pygame.joystick.Joystick(0)
    controller.init()

    return controller
        
def run_test(ui):
    arm = arm_movements.ArmMovements()
    config.DEAD_LETTER_LIST = config.make_dead_letter_list(config.COND2_ANSWER)
    # arm.reset_blocks()
    assist = assistance.get_assistance(input = True, utterance = True, ui = ui)
    reset_block_list = []
    letters_placed = {'1': 'E', '2': 'L', '3': 'A', '4': 'T', '5': 'S'}
    for slot, letter in letters_placed.items():
        reset_block_list.append([slot, letter, config.SLOT_POS[slot], config.SLOT_POS[letter]])
    print("reset_block_list:", reset_block_list)
    arm.reset_blocks(reset_block_list, assist)

def log(demo, assist, arm):
    if not demo:
        inference_log = assist.get_log()
        if not inference_log.empty:
            inference_log = inference_log.to_string()
            with open(config.INFERENCE_LOG_FILE, 'a') as file:
                file.write(inference_log)
                file.write('\n\n')

        robot_state_log = arm.get_robot_state_log()
        if not robot_state_log.empty:
            robot_state_log = robot_state_log.to_string()
            with open(config.ROBOT_STATE_LOG_FILE, 'a') as file:
                file.write(robot_state_log)
                file.write('\n\n')

        letter_data_log = arm.get_letter_data_log()
        if not letter_data_log.empty:
            letter_data_log = letter_data_log.to_string()
            with open(config.LETTER_DATA_LOG_FILE, 'a') as file:
                file.write(letter_data_log)
                file.write('\n\n')

        utterance_log = assist.get_ui_log()
        if not utterance_log.empty:
            utterance_log = utterance_log.to_string()
            with open(config.UTTERANCE_LOG_FILE, 'a') as file:
                file.write(utterance_log)
                file.write('\n\n')

    print("[STUDY MESSAGE] Teleoperation stopped.")
    rest_position()
    pygame.quit()
    print("DONE SHUTTING DOWN TELEOPERATION")

def run_free_teleop(controller, phase, assist=None, demo=False):
    try:
        print("[STUDY MESSAGE] Controller connected! Use the right stick to move, left stick up to pick block and left stick down to place block.")
        arm = arm_movements.ArmMovements()
        arm.init_teleoperation()

        while True:
            # TODO: LOG Can keep track of time taken to place letters/words here
            if config.NUM_LETTERS_PLACED == 5 and not config.DONE:
                config.RESETTING = True
                # Wait for 3 seconds to show the answer
                time.sleep(3)
                done = reset(arm, controller, phase, assist)
                arm.rest_position()
                arm.init_teleoperation()
                config.RESETTING = False
                if done:
                    config.DONE = True

            pygame.event.pump()  # Refresh pygame events
            arm.teleoperation(controller, phase, assist)
            time.sleep(0.1)  # Polling delay
    except KeyboardInterrupt or Exception as e:
        config.DONE = True
        log(demo, assist, arm)
    
# Checks block_positions list which stores (letter, position) tuples and returns blocks back
def reset(arm, controller, phase, assist):
    # try:
    reset_block_list = []
    print('letters_placed:', config.LETTERS_PLACED)
    for slot, letter in config.LETTERS_PLACED.items():
        reset_block_list.append([slot, letter, config.SLOT_POS[slot], config.SLOT_POS[letter]])
    print("reset_block_list:", reset_block_list)
    return arm.reset_blocks(reset_block_list, assist) == 5
    
# No Assistance
def run_condition_1(ui):
    # Construct the dead letter list
    answer = config.COND1_ANSWER
    config.DEAD_LETTER_LIST = config.make_dead_letter_list(answer)
    config.construct_answer_pos(answer)
    controller = init_controller()
    assistance_agent = assistance.get_assistance(input = False, utterance=False, ui = ui)
    run_free_teleop(controller, 'cond1', assistance_agent)

# Assistance with goal inference using controller input
def run_condition_2(ui):
    # Construct the dead letter list 
    answer = config.COND2_ANSWER
    config.DEAD_LETTER_LIST = config.make_dead_letter_list(answer)
    config.construct_answer_pos(answer)
    controller = init_controller()
    assistance_agent = assistance.get_assistance(input = True, utterance = False, ui = ui)
    run_free_teleop(controller, 'cond2', assistance_agent)

# Assistance with goal inference using verbalization
def run_condition_3(ui):
    # Construct the dead letter list
    answer = config.COND3_ANSWER
    config.DEAD_LETTER_LIST = config.make_dead_letter_list(answer)
    config.construct_answer_pos(answer)
    controller = init_controller()
    assistance_agent = assistance.get_assistance(input = False, utterance = True, ui = ui)
    run_free_teleop(controller, 'cond3', assistance_agent)

# Assistance with goal inference using controller input and verbalization
def run_condition_4(ui):
    # Construct the dead letter list
    answer = config.COND4_ANSWER
    config.DEAD_LETTER_LIST = config.make_dead_letter_list(answer)
    config.construct_answer_pos(answer)
    controller = init_controller()
    assistance_agent = assistance.get_assistance(input = True, utterance = True, ui = ui)
    run_free_teleop(controller, 'cond4', assistance_agent)

def cond_runners(cond, ui, participant_id):
    match cond:
        case 'cond1':
            # Initialize logger
            init_logger(participant_id, cond)
            # Connect and initialize arm
            rest_position()
            # Run condition 1
            run_condition_1(ui)
        case 'cond2':
            # Initialize logger
            init_logger(participant_id, cond)
            # Connect and initialize arm
            rest_position()
            # Run condition 2
            run_condition_2(ui)
        case 'cond3':
            # Initialize logger
            init_logger(participant_id, cond)
            # Connect and initialize arm
            rest_position()
            # Run condition 3
            run_condition_3(ui)
        case 'cond4':
            # Initialize logger
            init_logger(participant_id, cond)
            # Connect and initialize arm
            rest_position()
            # Run condition 4
            run_condition_4(ui)

def main_runner(participant_id, phase, training, ui):
    if training:
        # Initialize logger
        if phase == 'run1':
            init_logger(participant_id, 'run1_training')
        if phase == 'run2':
            init_logger(participant_id, 'run2_training')
        # Connect and initialize arm
        rest_position()
        # Free teleoperation
        controller = init_controller()
        run_free_teleop(controller, phase, ui)

    else:
        match phase:
            case 'demo':
                rest_position()
                # Free teleoperation
                controller = init_controller()
                assistance_agent = assistance.get_assistance(input = False, utterance = False, ui = ui)
                run_free_teleop(controller, phase, assistance_agent, demo=True)
            case 'test':
                # Connect and initialize arm
                rest_position()
                run_test(ui)
                # Bring arm back to initial position
                rest_position()
            case 'run1':
                cond_runners(config.PARTCIPANT_COND_DICT[participant_id][0], ui, participant_id)
            case 'run2':
                cond_runners(config.PARTCIPANT_COND_DICT[participant_id][1], ui, participant_id)
            case 'reset':
                # Connect and initialize arm
                rest_position()

def main():
    options = ['demo', 'training', 'run1', 'run2', 'reset', 'test']
    # Check if the phase was provided
    training = False

    if len(sys.argv) != 3 or sys.argv[2] not in options or not sys.argv[1].isnumeric():
        if len(sys.argv) == 4:
            if not isinstance(sys.argv[3], bool):
                print("Usage: python3 main.py <participant id> <phase> (Options: run1, run2, reset, test) <training> (Options: True or False, default is False)")
                sys.exit(1)
            else:
                training = sys.argv[3]
        else:
            print("Usage: python3 main.py <participant id> <phase> (Options: run1, run2, reset, test) <training> (Options: True or False, default is False)")
            sys.exit(1)
    
    # Get the phase from the command line argument
    participant_id = int(sys.argv[1])
    phase = sys.argv[2]

    ui = utterance_inference.UtteranceInference()
    main_runner(participant_id, phase, training, ui)

    try:
        # Main program logic here
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("CTRL+C pressed. Exiting...")
    finally:
        ui.stop_threads()
        print("Program stopped.")

if __name__ == "__main__":
    main()