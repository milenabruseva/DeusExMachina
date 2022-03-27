import os
import datetime
import glob
import json
import numpy as np

from ..features import state_dict_to_feature_str, store_unrecoverable_infos_helper
from ..parameter_decay import Explorer, PiggyCarry

from ..features import PreviousWinnerCD

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

        self.gamma = params["reward_decay"]
        self.explorer_type = params["explorer"]
        self.exp_param = params["exp_param"]
        self.exp_name = params["piggy_name"]

        self.upd_type = params["update_type"]
        self.upd_param = params["update_param"]

        self.learning_type = params["learning_type"]
        self.learning_param = params["learning_param"]
        self.save_n_rounds = params["save_n_rounds"]
        self.train_fast = params["train_fast"]

        self.event_reward_set = params["event_reward_set"]
        self.dyn_rewards = params["dyn_rewards"]

        self.feature = params["feature"]
        self.q_table_id = params["q_table_id"]
    self.q_table = {}
    self.n_table = {}
    if self.explorer_type != "piggyback":
        self.explorer = Explorer(self.explorer_type, self.exp_param)
    else:
        self.piggy = PiggyCarry(self.logger, self.exp_name, self.exp_param)

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

    # Long/Short Term Memory
    self.prev_game_state_str = None
    self.next_game_state_str = None
    self.prev_game_state = None
    self.remaining_coins_old = 9
    self.remaining_coins_new = 9
    self.killed_opponents_scores = {}
    self.own_bomb_old = None
    self.own_bomb_new = None

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

    pure_game_state = game_state

    # Reset at game start:
    if game_state["step"] == 1:
        self.prev_game_state_str = None
        self.next_game_state_str = None
        self.prev_game_state = None
        self.remaining_coins_old = 9
        self.remaining_coins_new = 9
        self.killed_opponents_scores = {}
        self.own_bomb_old = None
        self.own_bomb_new = None

    # Store unrecoverable game state information
    if self.feature in ["PreviousWinner", "PreviousWinnerCD"]:
        if self.prev_game_state is not None:
            self.remaining_coins_old = self.remaining_coins_new
            self.own_bomb_old = self.own_bomb_new

            coin_diff, killed_opponents_w_scores = store_unrecoverable_infos_helper(self.prev_game_state, game_state)

            self.remaining_coins_new -= coin_diff
            self.killed_opponents_scores.update(killed_opponents_w_scores)
            if self.own_bomb_old is None:
                if game_state["self"][3] in [bomb[0] for bomb in game_state["bombs"]]:
                    self.own_bomb_new = game_state["self"][3]
            else:
                if self.own_bomb_old not in [bomb[0] for bomb in game_state["bombs"]]:
                    self.own_bomb_new = None

        self.prev_game_state = game_state
        game_state["remaining_coins"] = self.remaining_coins_new
        game_state["own_bomb"] = self.own_bomb_new


    if self.next_game_state_str is not None:
        state_str = self.next_game_state_str
    else:
        state_str = state_dict_to_feature_str(game_state, self.feature)

    # Symmetry check
    transform = None
    if type(state_str) is not str:
        if self.feature not in ["RollingWindow", "PreviousWinnerCD"]:
            self.logger.warn("Non-single-state-string only implemented for RollingWindow yet.")
        state_idx = check_state_exist_w_sym(self, state_str[0])
        if state_idx is None or state_str[1][0] is None:
            # Just use the first entry w/o transform
            state_str = state_str[0][0]
        else:
            # Use transformed
            transform = state_str[1][state_idx]
            state_str = state_str[0][state_idx]

    check_state_exist_and_add(self, state_str)

    # action selection
    if self.explorer_type != "piggyback":
        action = self.explorer.explore(self.actions, self.q_table[state_str], self.n_table[state_str])
    else:
        action = ACTIONS.index(self.piggy.carry(game_state))

    if time_for_bed(game_state, self):
        action = ACTIONS.index("BOMB")

    # Transform action
    if self.explorer_type != "piggyback" and transform is not None:
        if not (action >= 4):  # wait or bomb
            transformed_action_layout = self.action_layout
            if transform[1]:
                transformed_action_layout = transformed_action_layout.T
            transformed_action_layout = np.rot90(transformed_action_layout, k=(-1) * transform[0])
            action = action_layout_to_action(self, transformed_action_layout, action)

    if not self.train_fast:
        print("In act()")
        PreviousWinnerCD(game_state).print_me()
        qs = self.q_table[state_str]
        qviz = np.array([[np.NAN, qs[0], np.NAN],
                         [qs[3], qs[4], qs[1]],
                         [np.NAN, qs[2], qs[5]]])
        if transform is not None:
            if transform[1]:
                qviz = qviz.T
            qviz = np.rot90(qviz, k=(-1) * transform[0])
        print(qviz)
        print("Action took: "+ACTIONS[action])

    # Save state string and transform in short term memory
    self.prev_game_state_str = ([state_str], [transform])

    return ACTIONS[action]


def time_for_bed(game_state, self):
    self_score = game_state["self"][1]
    remaining_points_possible = self.remaining_coins_new + len(game_state["others"]) * 5
    max_opponent_scores_possible = {}

    for opp in game_state["others"]:
        max_opponent_scores_possible[opp[0]] = opp[1] + remaining_points_possible - 5

    killed_opponents_winning = len({opponent:score for (opponent,score) in self.killed_opponents_scores.items() if score >= self_score})
    living_opponents_can_win = len({opponent:score for (opponent,score) in max_opponent_scores_possible.items() if score >= self_score})

    if self_score >= 10:
        return True
    elif not killed_opponents_winning and not living_opponents_can_win:
        return True
    elif remaining_points_possible == 0: # killed opponents are winning, but no more points possible
        return True
    else:
        return False


def check_state_exist_and_add(self, state_str):
    if state_str not in self.q_table:
        # append new state to tables
        if self.feature in ["PreviousWinnerCD"] and False:
            self.q_table[state_str] = [8, 8, 8, 8, 7.9, 7.9]
            for i in range(4):
                if state_str[i] == "1" :
                    self.q_table[state_str][i] = 10
                elif state_str[i] == "2":
                    self.q_table[state_str][i] = 0
            if state_str[5] in ["0", "1", "2"]:
                if state_str[:4].count("2") == 2 and ((state_str[:4].count("22") == 1) or (state_str[0] == "2" and state_str[3] == "2")):
                    self.q_table[state_str][5] = -10
                else:
                    if state_str[4] == "0":
                        pass
                    elif state_str[4] == "1":
                        self.q_table[state_str][5] = 10
                    elif state_str[4] == "2":
                        self.q_table[state_str][5] = 15
                    elif state_str[4] == "3":
                        self.q_table[state_str][5] = 20
                    elif state_str[4] == "4":
                        self.q_table[state_str][4] = -10
                        self.q_table[state_str][5] = -10
        else:
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
