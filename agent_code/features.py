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
    elif feature_type == "PreviousWinner":
        return PreviousWinner(state_dict)
    else:
        return None

def state_dict_to_feature_str(state_dict, feature_type):
    if feature_type == "LocalVision":
        return str(LocalVision(state_dict))
    elif feature_type == "RollingWindow":
        return RollingWindow(state_dict).get_all_sym_str()
    elif feature_type == "PreviousWinner":
        return PreviousWinner(state_dict).get_all_sym_str()
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


# Helper functions for previous year winner features
def manhattan_distance(x1: tuple[int, int], x2: tuple[int, int]):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])


# nearest entity
def nearest_entity_distance(coord, entities):
    if len(entities):
        min_distance = 99
        nearest_entity = (0, 0)
        for entity_pos in entities:
            dist = manhattan_distance(coord, entity_pos)
            if dist < min_distance:
                min_distance = dist
                nearest_entity = entity_pos
        return min_distance, nearest_entity
    else:
        return 0, coord


def get_neighbor_coords(coord):
    up = coord[0], coord[1] - 1
    right = coord[0] + 1, coord[1]
    down = coord[0], coord[1] + 1
    left = coord[0] - 1, coord[1]
    return [up, right, down, left]


def surrounding_blowable_count(coord, arena, enemy_locations):
    blowable_count = 0
    if coord[0] != 0 and coord[0] != 16 and coord[1] != 0 and coord[1] != 16:
        neighboring_tiles = get_neighbor_coords(coord)
        for tile in neighboring_tiles:
            if arena[tile] == 1 or tile in enemy_locations:  # if neighboring tile is crate or enemy agent
                blowable_count += 1

    return blowable_count

def safe_death(coord, blast_coords, game_mode):
    if game_mode == 2 and coord not in blast_coords:
        return True
    return False

def possibly_yields_danger(coord, nearest_enemy, arena):
    #if distance to nearest enemy is less than 4, and moving there would put player agent in a corner
    dist = manhattan_distance(coord, nearest_enemy)
    if coord[0] != 0 and coord[0] != 16 and coord[1] != 0 and coord[1] != 16:
        neighboring_tiles = get_neighbor_coords(coord)
        surrounding_solids = 0
        for tile in neighboring_tiles:
            if arena[tile] == 1 or arena[tile] == -1:
                surrounding_solids += 1
        if dist < 4 and surrounding_solids >= 3:
            return True
    return False

def get_neighboring_tile_info(coord, idx, coins, bombs, arena, enemy_locations, game_mode, nearest_tile_to_coin,
                              nearest_tile_to_enemy, most_destructive_tile, yields_safe_death, is_dangerous):
    if coord in bombs or arena[coord] == -1 or arena[coord] == 1 or coord in enemy_locations:  # if bomb/wall/crate/opponent on tile
        return 2

    # if tile yields safe death
    elif yields_safe_death[idx]:
        return 2

    elif is_dangerous[idx]: # if tile possibly yields danger, that is, its trapped on 3 sides with not enough time to escape
        return 3

    # if field is empty (or coin), no danger/safe death condition, choose mode dependent value
    elif arena[coord] == 0 or coord in coins:
        if game_mode == 0:  # coin collecting mode
            # if this tile is nearer to a coin than other neighboring tiles
            if coord == nearest_tile_to_coin:
                return 1
            else:
                return 0
        elif game_mode == 1:  # bombing/crate destroying mode
            # if planting a bomb on this tile would destroy more crates/opponents that current tile or other neighboring tiles
            # OR if value for neighboring tiles is still 0 and this tile is nearer to the nearest crate
            if coord == most_destructive_tile:
                return 1
            else:
                return 0

        elif game_mode == 2:  # terminator mode
            # if this tile is nearer to the nearest opponent than other neighboring tiles but not within their bomb spread
            if coord == nearest_tile_to_enemy:  # To-Do - bomb spread
                return 1
            else:
                return 0

    else:  # tile is free
        return 0


def get_current_tile_info(coord, bombs, blast_coords, game_mode, arena, enemy_locations):
    # if bomb placement would not lead to own self death
    if not safe_death(coord, blast_coords, game_mode):
        if surrounding_blowable_count(coord, arena, enemy_locations) >= 6: #destroy at least 6 crates/opponents
            return 3
        if surrounding_blowable_count(coord, arena, enemy_locations) >= 3: #destroy at least 3 crates/opponents
            return 2
        if surrounding_blowable_count(coord, arena, enemy_locations) >= 1: #destroy at least 1 crates/opponents
            return 1
    if coord in bombs or coord in blast_coords: #bomb on current tile or waiting leads to own safe death
        return 4
    else: #if agent does not die from waiting/ placing a bomb would result in own safe death/ not destroy crate
        return 0



def get_mode(coins_available, coins_remaining):
    if coins_available > 0:  # if no. visible coins is greater or equal to 1
        return 0  # coin collecting mode
    elif coins_remaining > 0:  # if no. visible coins is 0
        return 1  # bombing/crate destroying mode
    else:  # if enemy agents still alive and all coins have been collected
        return 2  # terminator mode

def coin_difference(old_game_state, new_game_state):
    if old_game_state is not None and new_game_state is not None:
        coins_old = set(old_game_state["coins"])
        coins_new = set(new_game_state["coins"])

        opponents_old = set(old_game_state["others"])
        opponents_new = set(new_game_state["others"])
        opponents_killed = opponents_old.difference(opponents_new)
        opponents_scores = []
        for i in opponents_killed:
            opponents_scores.append(i[1])

        return len(coins_old.difference(coins_new)), opponents_scores
    return 0, []


class PreviousWinner(Features):
    # features used by 'No Time For Caution'
    __slots__ = "features"
    features : list[int]

    def __init__(self, game_state: dict):
        # Relevant Game State Information
        arena = np.array([game_state["field"]])[0]
        coins = game_state["coins"]
        bombs = []
        bombs_countdown = []
        if len(game_state["bombs"]):
            for i in range(len(game_state["bombs"])):
                bombs.append(game_state["bombs"][i][0])
                bombs_countdown.append(game_state["bombs"][i][1])
        blast_coords = []
        blast_countdowns = []
        for idx, bomb_coord in enumerate(bombs):
            for coord in get_blast_coords(bomb_coord, arena):
                if coord not in blast_coords:
                    blast_coords.append(coord)
                    blast_countdowns.append(bombs_countdown[idx] + 1)
                else:
                    idxx = blast_coords.index(coord)
                    blast_countdowns[idxx] = min(blast_countdowns[idxx], bombs_countdown[idx] + 1)

        # player agent location
        location = np.array([game_state["self"][3][0], game_state["self"][3][1]])
        origin_xy = location[0], location[1]

        # neighbouring tiles
        neighboring_tiles = get_neighbor_coords(origin_xy)

        # enemy locations
        enemy_locations = []
        if len(game_state["others"]):
            for i in range(len(game_state["others"])):
                enemy_locations.append(game_state["others"][i][3])

        self.features = []
        game_mode = get_mode(len(game_state["coins"]), game_state["remaining_coins"])  # game mode feature

        # find neighboring tile nearest to coin and neighboring tile nearest to enemy agent
        nearest_coin_distance = 99
        nearest_enemy_distance = 99
        nearest_tile_to_coin = origin_xy
        nearest_tile_to_enemy = origin_xy

        for coord in neighboring_tiles:
            coin_dist, nearest_coin = nearest_entity_distance(coord, coins)
            enemy_dist, nearest_enemy = nearest_entity_distance(coord, enemy_locations)
            if coin_dist < nearest_coin_distance:
                nearest_coin_distance = coin_dist
                nearest_tile_to_coin = coord
            if enemy_dist < nearest_enemy_distance:
                nearest_enemy_distance = enemy_dist
                nearest_tile_to_enemy = coord

        # find nearest coin and enemy to current position
        nearest_coin_distance, nearest_coin = nearest_entity_distance(origin_xy, coins)
        nearest_enemy_distance, nearest_enemy = nearest_entity_distance(origin_xy, enemy_locations)

        # find neighboring tile that would do the most blowing up damage
        max_blow_count = surrounding_blowable_count(origin_xy, arena, enemy_locations)
        most_destructive_tile = origin_xy

        for coord in neighboring_tiles:
            blow_count = surrounding_blowable_count(coord, arena, enemy_locations)
            if blow_count > max_blow_count:
                max_blow_count = blow_count
                most_destructive_tile = coord

        if max_blow_count == 0:
            nearest_crate_distance = 0
            crates = np.argwhere(arena == 1)
            for coord in neighboring_tiles:
                crate_dist, nearest_crate = nearest_entity_distance(coord, crates)
                if crate_dist < nearest_crate_distance:
                    nearest_crate_distance = crate_dist
                    most_destructive_tile = coord

        # find which neighboring tile, if any, yields safe death
        yields_safe_death = [False for i in range(4)]
        for idx, coord in enumerate(neighboring_tiles):
            if safe_death(coord, blast_coords, game_mode):
                yields_safe_death[idx] = True

        # find which neighboring tile, if any, yields possible danger
        is_dangerous = [False for i in range(4)]
        for idx, coord in enumerate(neighboring_tiles):
            if possibly_yields_danger(coord, nearest_enemy, arena):
                is_dangerous[idx] = True

        # information on neighboring fields
        for idx, coord in enumerate(neighboring_tiles):
            self.features.append(
                get_neighboring_tile_info(coord, idx, coins, bombs, arena, enemy_locations, game_mode, nearest_tile_to_coin,
                                          nearest_tile_to_enemy, most_destructive_tile, yields_safe_death, is_dangerous))

        # information on current field
        self.features.append(get_current_tile_info(origin_xy, bombs, blast_coords, game_mode, arena, enemy_locations))  # tile self

        # game mode
        self.features.append(game_mode)

        print(np.array([[0, self.features[0], 0],
                        [self.features[3], self.features[4], self.features[1]],
                        [0, self.features[2], self.features[5]]]))

    def __repr__(self):
        return ''.join(str(e) for e in self.features)


    def get_all_sym_str(self) -> tuple[list[str], list[tuple[int, int]]]:
        feature_as_matrix = np.array([self.features[0:2], self.features[2:4]])
        transforms = []
        sym_str = []

        for rot in range(4):
            for flip in [0, 1]:
                transforms.append((rot, flip))

                transformed_matrix = np.rot90(feature_as_matrix, k=rot)
                if flip:
                    transformed_matrix = transformed_matrix.T
                sym_str.append(''.join(str(e) for e in transformed_matrix.flatten()) + ''.join(str(e) for e in self.features[4:]))

        return sym_str, transforms
