import os
import pickle
import random

import numpy as np
import sys
from collections import deque
import time

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = {
    'nearest safe spot dir': 'dir',
    'nearest coin dir': 'dir',
    'nearest crate dir': 'dir',
    'safe to bomb': 'bool',
    'nearest enemy dir': 'dir',
    'enemy is trapped': 'bool',
    'bomb available': 'bool'
}
PATH = {
    0: 'not available',
    1: 'current field',
    2: 'down',
    3: 'up',
    4: 'right',
    5: 'left'
}

def print_features(features):
    i = 0
    for key in FEATURES:
        res = None
        if FEATURES[key] == 'dir':  res = PATH[features[i]]
        if FEATURES[key] == 'bool': res = 'true' if features[i] == 1 else 'false'
        print(f'{key}:  {res}')
        i += 1
    print('******************************')

def model_evaluation(model, features):
    current = model
    for f in features:
        current = current[f]
    return current

def setup(self):
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((6, 6, 6, 2, 6, 2, 2, 6))

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        
def act(self, game_state: dict) -> str:

    if not game_state:
        return 'WAIT'

    random_prob = .05
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)
    
    features = state_to_features(game_state)
    #print_features(features)

    action = ACTIONS[np.argmax(model_evaluation(self.model, features))]
    #action = rule_based(features)
    return action

def state_to_features(game_state: dict):
    features = []

    _, _, bomb_available, agent_pos = game_state['self']
    field = game_state['field']
    coins = game_state['coins']
    pos_x, pos_y = agent_pos

    # direction to nearest safe spot
    goal = lambda x, y: field_is_safe(game_state, x, y) == 'SAFE'
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal, 'SEMI-SAFE'))

    # direction to nearest coin
    goal = lambda x, y: (x, y) in coins
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal, 'SAFE'))

    # direction to nearest crate
    goal = lambda x, y: (field[x, y+1] == 1 or
                                field[x, y-1] == 1 or
                                field[x+1, y] == 1 or
                                field[x-1, y] == 1)
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal, 'SAFE'))

    # safe to bomb 
    goal = lambda x, y: field_is_safe(game_state, x, y, pos_x, pos_y) == 'SAFE'
    features.append(int(shortest_path(game_state, field, pos_x, pos_y, goal, 'SEMI-SAFE', max_len=4) != 0))

    # nearest enemy direction
    goal = lambda x, y: enemy_in_explosion_range(game_state, x, y, only_custom_bomb=True)
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal, 'SAFE'))

    # enemy is trapped:
    goal = lambda x, y: field_is_safe(game_state, x, y, pos_x, pos_y) == 'SAFE'
    enemy_is_trapped = False
    for _, _ , _, pos in game_state['others']:
        x_e, y_e = pos
        if shortest_path(game_state, field, x_e, y_e, goal, 'SEMI-SAFE', max_len=4) == 0:
            enemy_is_trapped = True
            break
    features.append(int(enemy_is_trapped))

    # bomb available
    features.append(int(bomb_available))
    return features

def enemy_in_explosion_range(game_state, x, y, only_custom_bomb=False):
    for _, _ , _, pos in game_state['others']:
        x_e, y_e = pos
        if field_is_safe(game_state, x_e, y_e, x, y, only_custom_bomb=only_custom_bomb) == 'SEMI-SAFE':
            return True
    return False

def field_is_safe(game_state, pos_x, pos_y, bomb_x=None, bomb_y=None, only_custom_bomb=False):
    '''
    check if the given field is safe, ie: 
    there is no explosion and no explosion in the near future happening on this field
    '''
    field = game_state['field']
    bombs = game_state['bombs'].copy()
    if bomb_x and bomb_y:
        bombs.append(((bomb_x, bomb_y), 3))
    if only_custom_bomb:
        bombs = [((bomb_x, bomb_y), 3)]
    explosion_map = game_state['explosion_map']
    safe = 'SAFE'

    if explosion_map[pos_x, pos_y] != 0:
        safe = 'UNSAFE'

    for (x, y), t in bombs:
        if (pos_x == x and abs(y - pos_y) <= 3):
            s = 1 if y > pos_y else -1 
            wall = False
            for d in range(s, y-pos_y, s):
                if field[x, pos_y+d] == -1:
                    wall = True
            if not wall:
                safe = 'SEMI-SAFE'

        if (pos_y == y and abs(x - pos_x) <= 3):
            s = 1 if x > pos_x else -1 
            wall = False
            for d in range(s, x-pos_x, s):
                if field[pos_x+d, y] == -1:
                    wall = True
            if not wall:
                safe = 'SEMI-SAFE'

    return safe

def point_in_list(x, y, l):
    if len(l) == 0: return False
    return np.min(np.sum(abs(np.array(l)[:, :2] - [x, y]), axis=1)) == 0

def shortest_path(game_state, field, x_s, y_s, goal, path_type, max_len=np.inf):
    '''
    0: no path to goal
    1: at goal
    2, 3, 4, 5: goal is in (down, up, right, left) direction
    '''
    accepted_path_types = None
    if path_type == 'SAFE': accepted_path_types = ['SAFE']
    if path_type == 'SEMI-SAFE':    accepted_path_types = ['SAFE', 'SEMI-SAFE']
    if path_type == 'UNSAFE':   accepted_path_types = ['SAFE', 'SEMI-SAFE', 'UNSAFE']

    player_positions = [(x, y, -1) for _, _ , _, (x, y) in game_state['others']]
    _, _, _, (x, y) = game_state['self']
    player_positions.append((x, y, -1))

    fields_visited = []
    fields_to_check = deque([[x_s, y_s, None]])
    while fields_to_check:
        x, y, i = fields_to_check.popleft()
        
        if goal(x, y):
            i_current = i
            length = 0
            while True:
                if x == x_s and y == y_s:
                    return 1
                length += 1
                if length > max_len:
                    return 0
                if x == x_s and y == y_s+1:
                    return 2
                if x == x_s and y == y_s-1:
                    return 3
                if x == x_s+1 and y == y_s:
                    return 4
                if x == x_s-1 and y == y_s:
                    return 5
                x, y, i_current = fields_visited[i_current]

        fields_visited.append([x, y, i])
        i = len(fields_visited) - 1
        
        safe = field_is_safe(game_state, x-1, y) in accepted_path_types
        if field[x-1, y] == 0 and not point_in_list(x-1, y, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x-1, y, i])
        safe = field_is_safe(game_state, x+1, y) in accepted_path_types
        if field[x+1, y] == 0 and not point_in_list(x+1, y, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x+1, y, i])
        safe = field_is_safe(game_state, x, y-1) in accepted_path_types
        if field[x, y-1] == 0 and not point_in_list(x, y-1, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x, y-1, i])
        safe = field_is_safe(game_state, x, y+1) in accepted_path_types
        if field[x, y+1] == 0 and not point_in_list(x, y+1, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x, y+1, i])

    return 0


def rule_based(features):
    outputs = ['NONE', 'CURRENT', 'DOWN', 'UP', 'RIGHT', 'LEFT']
    safe_dir = outputs[features[0]]
    coin_dir = outputs[features[1]]
    crate_dir = outputs[features[2]]
    safe_to_bomb = bool(features[3])
    enemy_dir = outputs[features[4]]
    enemy_trapped = bool(features[5])
    bomb_available = bool(features[6])

    if safe_dir in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return safe_dir

    if enemy_trapped and bomb_available and safe_to_bomb:
        return 'BOMB'

    if coin_dir in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return coin_dir

    if (enemy_dir == 'CURRENT' or crate_dir == 'CURRENT') and bomb_available and safe_to_bomb:
        return 'BOMB'

    if crate_dir in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return crate_dir

    if enemy_dir in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return enemy_dir

    return 'WAIT'
    