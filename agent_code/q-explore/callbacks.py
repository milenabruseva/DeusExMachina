import os
import json

import numpy as np
import pandas as pd

from ..features import QFeatures

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup agent.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Setup actions and parameters
    self.actions = ACTIONS
    #self.actions = list(range(len(ACTIONS))) # integer representation of actions
    with open("q-exlore-params.json") as params_file:
        params = json.load(params_file)
        self.lr = params["learning_rate"]
        self.gamma = params["reward_decay"]
        self.exp = params["exploration_reward"]
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    self.n_table = pd.DataFrame(columns=self.actions, dtype=np.int64)

    # Load q table
    # check training: self.train
    if not os.path.isfile("q_table.json"):
        self.logger.info("No q_table.json found")
    else:
        self.logger.info("Loading q_table from json file")
        with open("q_table.json", "r") as q_table_file:
            q_table_json = json.load(q_table_file)
            self.q_table = pd.read_json(q_table_json, orient="index")

    # Load visit count
    if not os.path.isfile("n_table.json"):
        self.logger.info("No n_table.json found")
    else:
        self.logger.info("Loading n_table from json file")
        with open("n_table.json", "r") as n_table_file:
            n_table_json = json.load(n_table_file)
            self.n_table = pd.read_json(n_table_json, orient="index")


def act(self, game_state: dict) -> str:
    """
    Make a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state_str= str(state_to_features(game_state))
    check_state_exist(self,state_str)

    # action selection
    # choose best action
    state_action = self.q_table.loc[state_str, :]
    # some actions may have the same value, randomly choose one of these actions
    action = np.random.choice(state_action[state_action == np.max(state_action)].index)

    return action
    #return ACTIONS[action]


def check_state_exist(self, state_str):
    if state_str not in self.q_table.index:
        # append new state to q table
        self.q_table.loc[state_str] = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state_str,
            )
        self.n_table.loc[state_str] = pd.Series(
            [1] * len(self.actions),
            index=self.n_table.columns,
            name=state_str,
            )


def state_to_features(game_state: dict) -> QFeatures:
    """
    todo: maybe turn into function of Feature
    Conversion of game_state to features.
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    return QFeatures(game_state["self"][3][0], game_state["self"][3][1])
