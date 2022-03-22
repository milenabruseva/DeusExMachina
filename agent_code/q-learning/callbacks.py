import os
import datetime
import glob
import json
import numpy as np

import random

from ..features import state_dict_to_feature_str

ALGORITHM = 'q-learning'
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
    with open("q-learning-params.json") as params_file:
        self.logger.info("Loading q-learning parameters from json file")

        params = json.load(params_file)
        self.lr = params["learning_rate"]
        self.gamma = params["reward_decay"]
        self.epsilon = params["e_greedy"]
        self.event_reward_set = params["event_reward_set"]
        self.dyn_rewards = params["dyn_rewards"]
        self.train_fast = params["train_fast"]
        self.feature = params["feature"]
        self.q_table_id = params["q_table_id"]
        self.save_n_rounds = params["save_n_rounds"]
    self.q_table = {}

    # load q_table from json
    q_table_filenames = glob.glob('q_table*.json')
    self.proper_filename = ''
    last_q_table_dict = None
    for file in q_table_filenames:
        if not os.stat(file).st_size == 0:
            with open(file, "r") as q_table_file:
                last_q_table_dict = json.load(q_table_file)
                if "meta" in last_q_table_dict:
                    if last_q_table_dict["meta"]["algorithm"] == ALGORITHM and\
                            last_q_table_dict["meta"]["feature"] == self.feature and \
                            last_q_table_dict["meta"]["q_table_id"] == self.q_table_id:
                        self.proper_filename = file
                        break

    if not self.proper_filename == '':
        self.logger.info(f"Loading q_table from {self.proper_filename}")
        del last_q_table_dict["meta"]
        self.q_table = last_q_table_dict
    else:
        if not self.train:
            self.logger.warn("Not training and no q_table found")
        self.logger.info(f"No q_table*.json found, for feature {self.feature} and id {self.q_table_id}. Empty one is used")
        self.proper_filename = f'q_table_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'

   # Action layout for symmetry transformations
    self.action_layout = np.array([[6, self.ACTIONS.index("UP"), 6],
                                   [self.ACTIONS.index("LEFT"), 6, self.ACTIONS.index("RIGHT")],
                                   [6, self.ACTIONS.index("DOWN"), 6]], dtype=np.int8)


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
    if random.random() > self.epsilon:
        # choose best action
        q_values_of_state = self.q_table[state_str]
        # some actions may have the same value, randomly choose one of these actions
        action = random.choice([idx for idx, val in enumerate(q_values_of_state) if val == max(q_values_of_state)])
    else:
        # choose random action
        action = random.choice(self.actions)

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
        # append new state to q table
        self.q_table[state_str] = [0] * len(self.actions)


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
