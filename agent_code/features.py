from abc import abstractmethod
from typing import List, Tuple

import numpy as np

### Abstract Feature Base Class

class Features:
    @abstractmethod
    def __init__(self, game_state: dict):
        """
        Conversion of game_state to features.
        :param game_state:  A dictionary describing the current game board.
        :return: Feature
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Create unique string representation of features
        """
        pass


def state_dict_to_feature(state_dict, feature_type) -> Features:
    if feature_type == "LocalVision":
        return LocalVision(state_dict)
    elif feature_type == "RollingWindow":
        return RollingWindow(state_dict)
    else:
        return None

def state_dict_to_feature_str(state_dict, feature_type):
    if feature_type == "LocalVision":
        return str(LocalVision(state_dict))
    elif feature_type == "RollingWindow":
        return RollingWindow(state_dict).get_all_sym_str()
    else:
        return None

### Helper functions

def get_tile_type(coord, coins, bombs, arena):
    if coord in coins:
        return 2
    elif coord in bombs:
        return -1
    else:
        return arena[coord]


def get_blast_coords(bomb_pos, arena):
    x, y = bomb_pos[0], bomb_pos[1]
    blast_coords = [(x, y)]

    for i in range(1, 4):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, 4):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, 4):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, 4):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords


### Feature Implementations

# Plus shaped local vision
class LocalVision(Features):
    __slots__ = ("posx", "posy",
                 "left", "up", "right", "down", "origin",
                 "has_bomb")

    # Features
    posx: int
    posy: int

    left : int
    up : int
    right : int
    down : int
    origin : int

    has_bomb : bool

    def __init__(self, game_state: dict):
        bombs = []
        arena = np.array([game_state["field"]])[0]
        coins = game_state["coins"]
        if len(game_state["bombs"]):
            for i in range(len(game_state["bombs"])):
                bombs.append(game_state["bombs"][i][0])


        location = np.array([game_state["self"][3][0], game_state["self"][3][1]])
        self.posx = location[0]
        self.posy = location[1]
        self.vision = np.zeros(5)

        left_xy = location[0] - 1, location[1]
        up_xy = location[0], location[1] - 1
        right_xy = location[0] + 1, location[1]
        down_xy = location[0], location[1] + 1
        origin_xy = location[0], location[1]

        self.left = get_tile_type(left_xy, coins, bombs, arena)
        self.up = get_tile_type(up_xy, coins, bombs, arena)
        self.right = get_tile_type(right_xy, coins, bombs, arena)
        self.down = get_tile_type(down_xy, coins, bombs, arena)
        self.origin = get_tile_type(origin_xy, coins, bombs, arena)

        if game_state["self"][2]:
            self.has_bomb = True
        else:
            self.has_bomb = False

        #Search Coin
        #Nearest Opponent
        #Forseeable Explosion Danger

    def __repr__(self):
        return str(self.posx) + "|" + str(self.posy) + "|" + str(self.left) + "|" + str(self.up) + "|" +\
               str(self.right) + "|" + str(self.down) + "|" + str(self.origin) + "|" + str(self.has_bomb)


# 5x5 tiles Rolling Windows

def getWindowOrigin(player_location, arena):
    windows_center_x = player_x = player_location[0]
    windows_center_y = player_y = player_location[1]
    # check overlap left right
    overlap_left = 3 - player_x
    overlap_right = (arena.shape[0] - 4) - player_x
    overlap_up = 3 - player_y
    overlap_down = (arena.shape[1] - 4) - player_y

    if overlap_left > 0:
        windows_center_x += overlap_left
    elif overlap_right < 0:
        windows_center_x += overlap_right
    if overlap_up > 0:
        windows_center_y += overlap_up
    elif overlap_down < 0:
        windows_center_y += overlap_down

    return np.array([windows_center_x, windows_center_y])


def windows_to_str(vision, explosion):
    vision_str = ''.join(str(e) for e in vision.flatten())
    explosion_str = ''.join(str(e) for e in explosion.flatten())
    return vision_str + "|" + explosion_str


class RollingWindow(Features):
    # Features
    # vision : 5x5 grid tracing player showing all objects
    # free: 0, crate: 1, walls:2, player: 3, enemy: 4, coin: 5, bomb: 6, player_bomb: 7, enemy_bomb: 8
    # explosion_map : 5x5 grid tracing player showing explosions
    # no danger: 0, would_die_if_moved_onto/explosion: 1, in 1 round: 2, in 2 rounds: 3, in 3 rounds: 4, in 4 rounds: 5

    # todo: has bomb, nearest enemy outside of grid(direction, manhattan distance)

    def __init__(self, game_state: dict):
        # Relevant Game State Information
        arena = np.array([game_state["field"]])[0]
        coins = game_state["coins"]
        bombs_coords = []
        bombs_countdown = []
        if len(game_state["bombs"]):
            for i in range(len(game_state["bombs"])):
                bombs_coords.append(game_state["bombs"][i][0])
                bombs_countdown.append(game_state["bombs"][i][1])
        blast_coords = []
        blast_countdowns = []
        for idx, bomb_coord in enumerate(bombs_coords):
            for coord in get_blast_coords(bomb_coord, arena):
                if coord not in blast_coords:
                    blast_coords.append(coord)
                    blast_countdowns.append(bombs_countdown[idx] + 1)
                else:
                    idxx = blast_coords.index(coord)
                    blast_countdowns[idxx] = min(blast_countdowns[idxx], bombs_countdown[idx] + 1)
        explosion_map = game_state["explosion_map"]

        # Player bomb status
        self.has_bomb_left = game_state["self"][2]

        # Agent locations
        player_location = game_state["self"][3]
        enemy_locations = []
        if len(game_state["others"]):
            for i in range(len(game_state["others"])):
                enemy_locations.append(game_state["others"][i][3])

        # Windows
        self.vision = np.zeros((5,5), dtype=np.int8)
        self.explosion_map = np.zeros((5,5), dtype=np.int8)

        window_origin = getWindowOrigin(player_location, arena)

        window = np.array([[(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2)],
                    [(-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2)],
                      [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)],
                      [(1, -2), (1, -1), (1, 0), (1, 1), (1,2)],
                     [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]])

        tile_states = ["free", "crate", "walls", "player", "enemy", "coin", "bomb", "player_bomb", "enemy_bomb"]
        explosion_states = ["safe", "explosion", "1", "2", "3", "4", "5"]

        # Populate windows
        for i in range(self.vision.shape[0]):
            for j in range(self.vision.shape[1]):
                arena_xy = window_origin[0] + window[i, j][0], window_origin[1] + window[i, j][1]
                self.vision[i, j] = arena[arena_xy]

                # Fill vision
                if self.vision[i, j] == -1:
                    self.vision[i, j] = tile_states.index("walls")
                elif arena_xy in coins:
                    self.vision[i, j] = tile_states.index("coin")
                elif arena_xy in bombs_coords and arena_xy == player_location:
                    self.vision[i, j] = tile_states.index("player_bomb")
                elif arena_xy in bombs_coords and arena_xy in enemy_locations:
                    self.vision[i, j] = tile_states.index("enemy_bomb")
                elif arena_xy in bombs_coords:
                    self.vision[i, j] = tile_states.index("bomb")
                elif arena_xy == player_location:
                    self.vision[i, j] = tile_states.index("player")
                elif arena_xy in enemy_locations:
                    self.vision[i, j] = tile_states.index("enemy")

                if arena_xy in blast_coords:
                    self.explosion_map[i, j] = blast_countdowns[blast_coords.index(arena_xy)]
                if explosion_map[arena_xy] > 0:
                    self.explosion_map[i, j] = explosion_states.index("explosion")


    def __repr__(self):
        return str(int(self.has_bomb_left)) + "|" + windows_to_str(self.vision, self.explosion_map)


    def get_all_sym_str(self) -> tuple[list[str], list[tuple[int, int]]]:
        transforms = []
        sym_str = []

        for rot in range(4):
            for flip in [0, 1]:
                transforms.append((rot, flip))

                transformed_vision = np.rot90(self.vision, k=rot)
                transformed_explosion = np.rot90(self.explosion_map, k=rot)
                if flip:
                    transformed_vision = transformed_vision.T
                    transformed_explosion = transformed_explosion.T
                sym_str.append(str(int(self.has_bomb_left)) + "|" + windows_to_str(transformed_vision, transformed_explosion))

        return sym_str, transforms

