import os
import json

import numpy as np
import pandas as pd

from ..features import RollingWindow as LocalVision

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup agent.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.actions = ACTIONS
    with open("q-learning-params.json") as params_file:
        self.logger.info("Loading q-learning parameters from json file")
        params = json.load(params_file)
        self.lr = params["learning_rate"]
        self.gamma = params["reward_decay"]
        self.epsilon = params["e_greedy"]
        self.train_fast = params["train_fast"]
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # check training: self.train
    # load q_table from json
    # if self.train check
    if not os.path.isfile("q_table.json"):
        if not self.train:
            self.logger.warn("Not training and no q_table found")
        self.logger.info("No q_table.json found, empty one is used")
    else:
        self.logger.info("Loading q_table from json file")
        with open("q_table.json", "r") as q_table_file:
            q_table_json = json.load(q_table_file)
            self.q_table = pd.read_json(q_table_json, orient="index")


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

    state_str= str(LocalVision(game_state))
    check_state_exist(self,state_str)

    # action selection
    if np.random.uniform() > self.epsilon:
        # choose best action
        state_action = self.q_table.loc[state_str, :]
        # some actions may have the same value, randomly choose one of these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
    else:
        # choose random action
        action = np.random.choice(self.actions)
    return action


def check_state_exist(self, state_str):
    if state_str not in self.q_table.index:
        # append new state to q table
        self.q_table.loc[state_str] = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state_str,
            )
