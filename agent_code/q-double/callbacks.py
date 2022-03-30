import os
import json

import numpy as np
import pandas as pd

from agent_code.utils.features import QFeatures

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup agent.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.actions = ACTIONS
    #self.actions = list(range(len(ACTIONS))) # integer representation of actions
    with open("q-double-params.json") as params_file:
        params = json.load(params_file)
        self.lr = params["learning_rate"]
        self.gamma = params["reward_decay"]
        self.epsilon = params["e_greedy"]
    self.q1_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    self.q2_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # logger: logger.info("")
    # check training: self.train
    # load q_table from json
    # if self.train check
    if not os.path.isfile("q1_table.json") or not os.path.isfile("q2_table.json"):
        self.logger.info("Either q1_table.json or q2_table.json not found")
    else:
        self.logger.info("Loading q_tables from json file")
        with open("q1_table.json", "r") as q1_table_file:
            q1_table_json = json.load(q1_table_file)
            self.q1_table = pd.read_json(q1_table_json, orient="index")
        with open("q2_table.json", "r") as q2_table_file:
            q2_table_json = json.load(q2_table_file)
            self.q2_table = pd.read_json(q2_table_json, orient="index")


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
    if np.random.uniform() > self.epsilon:
        # choose best action from q1+q2
        state_action = self.q1_table.loc[state_str, :] + self.q2_table.loc[state_str, :]
        # some actions may have the same value, randomly choose one of these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
    else:
        # choose random action
        action = np.random.choice(self.actions)
    return action
    #return ACTIONS[action]


def check_state_exist(self, state_str):
    if state_str not in self.q1_table.index:
        # append new state to q table
        # self.q_table = self.q_table.concat(
        #     pd.Series(
        #         [0] * len(self.actions),
        #         index=self.q_table.columns,
        #         name=state_str,
        #     )
        # )
        self.q1_table.loc[state_str] = pd.Series(
                [0] * len(self.actions),
                index=self.q1_table.columns,
                name=state_str,
            )
        self.q2_table.loc[state_str] = pd.Series(
            [0] * len(self.actions),
            index=self.q2_table.columns,
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
