import os
import glob
import json
import random

import numpy as np

from .features import state_dict_to_feature_str, store_unrecoverable_infos_helper

ALGORITHM = 'q-forest'
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

    self.q_tables = []


    # Check for q/n_table and load q_table from json
    q_table_filenames = glob.glob('q_table*.json')
    for file in q_table_filenames:
        if not os.stat(file).st_size == 0:
            with open(file, "r") as q_table_file:
                q_table_temp = {}
                q_table_temp = json.load(q_table_file)
                del q_table_temp["meta"]
                self.q_tables.append(q_table_temp)


    # Action layout for symmetry transformations
    self.action_layout = np.array([[6, self.ACTIONS.index("UP"), 6],
                                   [self.ACTIONS.index("LEFT"), 6, self.ACTIONS.index("RIGHT")],
                                   [6, self.ACTIONS.index("DOWN"), 6]], dtype=np.int8)

    # Long/Short Term Memory
    self.prev_game_state_str = None
    self.next_game_state_str = None
    self.prev_game_state = None
    self.remaining_coins_old = 9
    self.remaining_coins_new = 9
    self.killed_opponents_scores = {}
    self.own_bomb_old = None
    self.own_bomb_new = None


def act(self, game_state: dict) -> str:
    """
    Make a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
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


    sym_state_strs = state_dict_to_feature_str(game_state, "DeusExMachinaFeatures")
    action_votes = [0] * len(self.actions)

    # Let q trees give their votes
    for q_table in self.q_tables:
        # Check if any equivalent state is in this q table
        index = check_state_exist_w_sym(sym_state_strs[0], q_table)
        if index is not None: # found
            existing_state_str = sym_state_strs[0][index]
            transform = sym_state_strs[1][index]

            # Get actions of maximal q value
            max_q_actions = [idx for idx, val in enumerate(q_table[existing_state_str]) if val == max(q_table[existing_state_str])]

            # transform actions back
            for idx in range(len(max_q_actions)):
                if not (max_q_actions[idx] >= 4):  # wait or bomb
                    transformed_action_layout = self.action_layout
                    if transform[1]:
                        transformed_action_layout = transformed_action_layout.T
                    transformed_action_layout = np.rot90(transformed_action_layout, k=(-1) * transform[0])
                    max_q_actions[idx] = action_layout_to_action(self, transformed_action_layout, max_q_actions[idx])

            # add count of actions
            for actions in max_q_actions:
                action_votes[actions] += 1

        else: # not found
            pass


    # Calculate valid actions
    # Gather information about the game state
    arena = game_state['field']
    player_pos = game_state['self'][3]
    bomb_pos = [bomb[0] for bomb in game_state['bombs']]
    enemy_pos = [enemy[3] for enemy in game_state['others']]

    x, y = player_pos[0], player_pos[1]
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_places, valid_actions, valid_movement_actions = [], [], []
    for neighbor in neighbors:
        if (arena[neighbor] == 0) and (neighbor not in bomb_pos) and (neighbor not in enemy_pos):
            valid_places.append(neighbor)
    if (x - 1, y) in valid_places:
        valid_actions.append(ACTIONS.index('LEFT'))
        valid_movement_actions.append(ACTIONS.index('LEFT'))
    if (x + 1, y) in valid_places:
        valid_actions.append(ACTIONS.index('RIGHT'))
        valid_movement_actions.append(ACTIONS.index('RIGHT'))
    if (x, y - 1) in valid_places:
        valid_actions.append(ACTIONS.index('UP'))
        valid_movement_actions.append(ACTIONS.index('UP'))
    if (x, y + 1) in valid_places:
        valid_actions.append(ACTIONS.index('DOWN'))
        valid_movement_actions.append(ACTIONS.index('DOWN'))
    valid_actions.append(ACTIONS.index('WAIT'))
    if game_state['step'] != 1:
        valid_actions.append(ACTIONS.index('BOMB'))


    # Compute most voted valid action
    #print(action_votes)

    popular_actions = [idx for idx, val in enumerate(action_votes) if val == max(action_votes)]
    valid_popular_actions = []
    for action in popular_actions:
        if action in valid_actions:
            valid_popular_actions.append(action)


    # if sum(action_votes) == 0:
    #     print("New state encountered.")

    if len(valid_popular_actions) > 0 and sum(action_votes) > 0:
        action = random.choice(valid_popular_actions)
    elif len(valid_movement_actions) > 0:
        action = random.choice(valid_movement_actions)
    else:
        action = ACTIONS.index('WAIT')

    if time_for_bed(game_state, self):
        action = ACTIONS.index("BOMB")

    return ACTIONS[action]


def check_state_exist_w_sym(state_str_list: list[str], q_table: dict):
    idx = None
    for i, state_str in enumerate(state_str_list):
        if state_str in q_table:
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