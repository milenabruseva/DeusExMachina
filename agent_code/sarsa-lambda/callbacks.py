import os
import datetime
import glob
import json
import numpy as np

import random

from ..features import state_dict_to_feature_str
from ..parameter_decay import Explorer

ALGORITHM = 'sarsa-lambda'
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup agent.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.ACTIONS = ACTIONS
    self.actions = range(len(ACTIONS))
    with open("sarsa-lambda-params.json") as params_file:
        self.logger.info("Loading sarsa-lambda parameters from json file")

        params = json.load(params_file)

        self.gamma = params["reward_decay"]
        self.explorer_type = params["explorer"]
        self.exp_param = params["exp_param"]

        self.learning_type = params["learning_type"]
        self.learning_param = params["learning_param"]
        self.lambda_ = params["trace_decay"]
        self.save_n_rounds = params["save_n_rounds"]
        self.train_fast = params["train_fast"]

        self.event_reward_set = params["event_reward_set"]
        self.dyn_rewards = params["dyn_rewards"]

        self.feature = params["feature"]
        self.q_table_id = params["q_table_id"]
    self.q_table = {}
    self.n_table = {}
    self.explorer = Explorer(self.explorer_type, self.exp_param)

    # Check for q/n_table and load q_table from json
    q_table_filenames = glob.glob('q_table*.json')
    self.q_table_filename = ''
    self.n_table_filename = ''
    last_q_table_dict = None
    for file in q_table_filenames:
        if not os.stat(file).st_size == 0:
            with open(file, "r") as q_table_file:
                last_q_table_dict = json.load(q_table_file)
                if "meta" in last_q_table_dict:
                    if last_q_table_dict["meta"]["algorithm"] == ALGORITHM and \
                            last_q_table_dict["meta"]["feature"] == self.feature and \
                            last_q_table_dict["meta"]["q_table_id"] == self.q_table_id:
                        self.q_table_filename = file

                        # Check n_table
                        n_table_filename_try = self.q_table_filename.replace("q", "n", 1)
                        with open(n_table_filename_try, "r") as n_table_file:
                            last_n_table_dict = json.load(n_table_file)
                            if "meta" in last_n_table_dict:
                                if last_n_table_dict["meta"]["algorithm"] == ALGORITHM and \
                                        last_n_table_dict["meta"]["feature"] == self.feature and \
                                        last_n_table_dict["meta"]["q_table_id"] == self.q_table_id:
                                    self.n_table_filename = n_table_filename_try
                                    break
                                else:
                                    self.logger.warn(f"q_table {self.q_table_filename} found without corresponding n_table.")

    if not (self.q_table_filename == '' or self.n_table_filename == ''):
        self.logger.info(f"Loading q_table from {self.q_table_filename} and n_table from {self.n_table_filename}")
        del last_q_table_dict["meta"]
        del last_n_table_dict["meta"]
        self.q_table = last_q_table_dict
        self.n_table = last_n_table_dict
    else:
        if not self.train:
            self.logger.warn("Not training and no q_table found")
        self.logger.info(f"No q_table*.json <-> n_table*.json pair found, for feature {self.feature} and id {self.q_table_id}. Empty one is used")
        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.q_table_filename = f'q_table_{date_time_str}.json'
        self.n_table_filename = f'n_table_{date_time_str}.json'

   # Action layout for symmetry transformations
    self.action_layout = np.array([[6, self.ACTIONS.index("UP"), 6],
                                   [self.ACTIONS.index("LEFT"), 6, self.ACTIONS.index("RIGHT")],
                                   [6, self.ACTIONS.index("DOWN"), 6]], dtype=np.int8)

    # Instantiate Eligibility Trace (dict of array with array[0] the action, array[1] the trace)
    self.eligibility_trace = {}


def act(self, game_state: dict) -> str:
    """
    Make a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    #state = LocalVision(game_state)
    #print(state.vision.T)
    #print(state.explosion_map.T)
    #print("-----------------------")

    state_str = state_dict_to_feature_str(game_state, self.feature)

    # Symmetry check
    transform = None
    if type(state_str) is not str:
        if self.feature != "RollingWindow":
            self.logger.warn("Non-single-state-string only implemented for RollingWindow yet.")
        state_idx = check_state_exist_w_sym(self, state_str[0])
        if state_idx is None:
            # Just use the first entry w/o transform
            state_str = state_str[0][0]
        else:
            # Use transformed
            transform = state_str[1][state_idx]
            state_str = state_str[0][state_idx]

    check_state_exist_and_add(self, state_str)

    # action selection
    action = self.explorer.explore(self.actions, self.q_table[state_str], self.n_table[state_str])

    # Transform action
    if transform is not None:
        if not (action >= 4):  # wait or bomb
            transformed_action_layout = self.action_layout
            if transform[1]:
                transformed_action_layout = transformed_action_layout.T
            transformed_action_layout = np.rot90(transformed_action_layout, k=(-1) * transform[0])
            action = action_layout_to_action(self, transformed_action_layout, action)

    return ACTIONS[action]


def check_state_exist_and_add(self, state_str):
    if state_str not in self.q_table:
        # append new state to tables
        self.q_table[state_str] = [0] * len(self.actions)
        self.n_table[state_str] = [1] * len(self.actions)


def check_state_exist_w_sym(self, state_str_list: list[str]):
    idx = None
    for i, state_str in enumerate(state_str_list):
        if state_str in self.q_table:
            idx = i
            break
    return idx


def action_layout_to_action(self, action_layout, action):
    act = action
    position = np.argwhere(action_layout == act)[0]

    if np.array_equal(position, [0,1]):
        act = self.ACTIONS.index("UP")
    elif np.array_equal(position, [1,0]):
        act = self.ACTIONS.index("LEFT")
    elif np.array_equal(position, [1,2]):
        act = self.ACTIONS.index("RIGHT")
    elif np.array_equal(position, [2,1]):
        act = self.ACTIONS.index("DOWN")
    else:
        self.logger.warn("Action layout to action transformation didn't find any valid action")
        act = None

    return act
