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
    elif feature_type == "PreviousWinnerCD":
        return PreviousWinnerCD(state_dict)
    else:
        return None

def state_dict_to_feature_str(state_dict, feature_type):
    if feature_type == "LocalVision":
        return str(LocalVision(state_dict))
    elif feature_type == "RollingWindow":
        return RollingWindow(state_dict).get_all_sym_str()
    elif feature_type == "PreviousWinner":
        return PreviousWinner(state_dict).get_all_sym_str()
    elif feature_type == "PreviousWinnerCD":
        return PreviousWinnerCD(state_dict).get_all_sym_str()
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

def places_reachable_from_coord(coord, arena, bombs, deadly, enemy_locations):
    places_reachable = []
    neighbors = get_neighbor_coords(coord)
    for pos in neighbors:
        if not ((arena[pos] in (1,-1)) or (pos in bombs) or (pos in deadly) or (pos in enemy_locations)):
            places_reachable.append(pos)

def update_deadly(blast_coords, blast_countdowns, explosions):
    deadly_coords = []
    for idx, coord in enumerate(blast_coords):
        if blast_countdowns[idx] == 0:
            deadly_coords.append(coord)
    deadly_coords.extend(tuple(map(tuple, np.argwhere(explosions == 1))))

    return deadly_coords


# nearest entity
def nearest_entity_distance(coord, entities):
    if len(entities):
        min_distance = 99
        nearest_entity_pos = (0, 0)
        for entity_pos in entities:
            dist = manhattan_distance(coord, entity_pos)
            if dist < min_distance:
                min_distance = dist
                nearest_entity_pos = entity_pos
        return min_distance, nearest_entity_pos
    else:
        return 0, coord


def get_neighbor_coords(coord):
    up = coord[0], coord[1] - 1
    right = coord[0] + 1, coord[1]
    down = coord[0], coord[1] + 1
    left = coord[0] - 1, coord[1]
    return [up, right, down, left]


def surrounding_blow_count(coord, arena, enemy_locations):
    blow_count = 0

    blast_coords = get_blast_coords(coord, arena)
    for blast_coord in blast_coords:
        if arena[blast_coord] == 1 or blast_coord in enemy_locations:
            blow_count += 1

    return blow_count

def safe_death(coord, blast_coords, game_mode):
    if game_mode == 2 and coord not in blast_coords:
        return True
    return False

def certain_death_in_n_steps(coord, arena, player_pos, bombs, blast_coords, blast_countdowns, explosions, enemy_locations, n=4):
    pass

def safe_to_bomb(player_pos, arena, has_bomb, deadly_coords):
    if has_bomb:
        safe_left = True
        safe_right = True
        safe_up = True
        safe_down = True
        x, y = player_pos
        for i in range(1, 4):
            if 0 <= x + i <= 16 and 0 <= y <= 16:
                if arena[(x+i, y)] != 0 or (x+i, y) in deadly_coords:
                    safe_right = False
            if 0 <= x - i <= 16 and 0 <= y <= 16:
                if arena[(x-i, y)] != 0 or (x-i, y) in deadly_coords:
                    safe_left = False
            if 0 <= x <= 16 and 0 <= y + i <= 16:
                if arena[(x, y+i)] != 0 or (x, y+i) in deadly_coords:
                    safe_up = False
            if 0 <= x <= 16 and 0 <= y - i <= 16:
                if arena[(x, y - i)] != 0 or (x, y-i) in deadly_coords:
                    safe_down = False

        if safe_left or safe_right or safe_down or safe_up:
            return True

        if not safe_right:
            for i in range(1, 3):
                if 0 <= x + i <= 16:
                    if 0 <= y + 1 <= 16:
                        if arena[(x + i, y + 1)] == 0 and (x + i, y + 1) not in deadly_coords:
                            return True
                    if 0 <= y - 1 <= 16:
                        if arena[(x + i, y - 1)] == 0 and (x + i, y - 1) not in deadly_coords:
                            return True
        if not safe_left:
            for i in range(1, 3):
                if 0 <= x - i <= 16:
                    if 0 <= y + 1 <= 16:
                        if arena[(x - i, y + 1)] == 0 and (x - i, y + 1) not in deadly_coords:
                            return True
                    if 0 <= y - 1 <= 16:
                        if arena[(x - i, y - 1)] == 0 and (x - i, y - 1) not in deadly_coords:
                            return True
        if not safe_up:
            for i in range(1, 3):
                if 0 <= y + i <= 16:
                    if 0 <= x + 1 <= 16:
                        if arena[(x + 1, y + i)] == 0 and (x + 1, y + i) not in deadly_coords:
                            return True
                    if 0 <= x - 1 <= 16:
                        if arena[(x - 1, y + i)] == 0 and (x - 1, y + i) not in deadly_coords:
                            return True
        if not safe_down:
            for i in range(1, 3):
                if 0 <= y - i <= 16:
                    if 0 <= x + 1 <= 16:
                        if arena[(x + 1, y - i)] == 0 and (x + 1, y - i) not in deadly_coords:
                            return True
                    if 0 <= x - 1 <= 16:
                        if arena[(x - 1, y - i)] == 0 and (x - 1, y - i) not in deadly_coords:
                            return True
        return False
    return False

def possibly_yields_danger(coord, arena, blast_coords, blast_countdowns, bombs, blast_to_bomb, enemy_locations):
    nearest_enemy_distance, nearest_enemy = nearest_entity_distance(coord, enemy_locations)

    #if distance to nearest enemy is less than 4, and moving there would put player agent in a corner
    neighboring_tiles = get_neighbor_coords(coord)
    surrounding_solids = 0
    for tile in neighboring_tiles:
        if arena[tile] in (1, -1):
            surrounding_solids += 1
    if nearest_enemy_distance < 4 and surrounding_solids >= 3:
        return True

    # Running towards a bomb
    if coord in blast_coords:
        idx = blast_coords.index(coord)
        bomb_pos = bombs[blast_to_bomb[idx]]
        dist_to_bomb = manhattan_distance(coord, bomb_pos)
        if (blast_countdowns[idx]) - (3 - dist_to_bomb) <= 0:
            return True

    return False

def get_neighboring_tile_info(coord, idx, coins, bombs, arena, enemy_locations, game_mode, nearest_tile_to_coin,
                              nearest_tile_to_enemy, most_destructive_tile, yields_certain_death, is_dangerous):

    if coord in bombs or arena[coord] == -1 or arena[coord] == 1 or coord in enemy_locations or yields_certain_death[idx]:  # if bomb/wall/crate/opponent/certainDeath on tile
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


def get_neighboring_tile_infos(player_pos, coords, coins, bombs, arena, enemy_locations, game_mode,
                              enemy_blast_coords, yields_certain_death, blast_coords, blast_countdowns,
                              blast_to_bomb):
    output = np.zeros(4, dtype=np.int8)

    for idx, coord in enumerate(coords):
        if coord in bombs or arena[coord] == -1 or arena[coord] == 1 or coord in enemy_locations or yields_certain_death[idx]:  # if bomb/wall/crate/opponent/certainDeath on tile
            output[idx] = 2
        elif possibly_yields_danger(coord, arena, blast_coords, blast_countdowns, bombs, blast_to_bomb, enemy_locations):
            output[idx] = 3

    # if field is empty (or coin), no danger/safe death condition, choose mode dependent value
    still_empty_idxs = np.where(output < 2)[0]
    if still_empty_idxs.size > 0:
        if game_mode == 0:  # coin collecting mode
            # find leftover tile nearest to any coin
            nearest_coin_distance = np.inf
            nearest_idx = []

            for idx in still_empty_idxs:
                coin_dist = nearest_entity_distance(coords[idx], coins)[0]
                if coin_dist < nearest_coin_distance:
                    nearest_coin_distance = coin_dist
                    nearest_idx.clear()
                    nearest_idx.append(idx)
                elif coin_dist == nearest_coin_distance:
                    nearest_idx.append(idx)

            output[np.random.choice(nearest_idx)] = 1
            # for idx in nearest_idx:
            #     output[idx] = 1


        elif game_mode == 1:  # bombing/crate destroying mode
            # if planting a bomb on this tile would destroy more crates/opponents that current tile or other neighboring tiles
            # OR if value for neighboring tiles is still 0 and this tile is nearer to the nearest crate

            max_blow_count = surrounding_blow_count(player_pos, arena, enemy_locations)
            max_blow_count_idx = [] # last entry if current field

            for idx in still_empty_idxs:
                blow_count = surrounding_blow_count(coords[idx], arena, enemy_locations)
                if blow_count > max_blow_count:
                    max_blow_count = blow_count
                    max_blow_count_idx.clear()
                    max_blow_count_idx.append(idx)
                elif blow_count == max_blow_count:
                    max_blow_count_idx.append(idx)

            if len(max_blow_count_idx) == 0:
                nearest_crate_distance = np.inf
                nearest_crate_distance_idx = []

                for idx in still_empty_idxs:
                    crates = np.argwhere(arena == 1)
                    #crates = tuple(map(tuple, crates))
                    crate_dist, _ = nearest_entity_distance(coords[idx], crates)
                    if crate_dist < nearest_crate_distance:
                        nearest_crate_distance = crate_dist
                        nearest_crate_distance_idx.clear()
                        nearest_crate_distance_idx.append(idx)
                    elif crate_dist == nearest_crate_distance:
                        nearest_crate_distance_idx.append(idx)

                max_blow_count_idx = nearest_crate_distance_idx

            output[np.random.choice(max_blow_count_idx)] = 1
            # for idx in max_blow_count_idx:
            #     output[idx] = 1


        elif game_mode == 2:  # terminator mode
            # if this tile is nearer to the nearest opponent than other neighboring tiles but not within their bomb spread

            # Check if field is in enemy bomb spread
            not_bomb_spread_idxs = []
            for idx in still_empty_idxs:
                if coords[idx] not in enemy_blast_coords:
                    not_bomb_spread_idxs.append(idx)

            if len(not_bomb_spread_idxs) > 0:
                nearest_opp_distance = np.inf
                nearest_idx = []

                for idx in not_bomb_spread_idxs:
                    opp_dist = nearest_entity_distance(coords[idx], enemy_locations)[0]
                    if opp_dist < nearest_opp_distance:
                        nearest_opp_distance = opp_dist
                        nearest_idx.clear()
                        nearest_idx.append(idx)
                    elif opp_dist == nearest_opp_distance:
                        nearest_idx.append(idx)

                output[np.random.choice(not_bomb_spread_idxs)] = 1
                # for idx in nearest_idx:
                #     output[idx] = 1

    return output


def get_current_tile_info(coord, can_bomb, bombs, deadly_coords, blast_coords, blast_countdowns, explosions, game_mode, arena, enemy_locations):

    if (coord in bombs) or (coord in deadly_coords): #bomb on current tile or staying on place leads to own certain death
        return 4
    elif can_bomb:
        if surrounding_blow_count(coord, arena, enemy_locations) >= 6: #destroy at least 6 crates/opponents
            return 3
        if surrounding_blow_count(coord, arena, enemy_locations) >= 3: #destroy at least 3 crates/opponents
            return 2
        if surrounding_blow_count(coord, arena, enemy_locations) >= 1: #destroy at least 1 crates/opponents
            return 1
        return 0
    else:
        return 0

def get_mode(visible_coins, coins_remaining, num_opps_left):
    if coins_remaining == 0 and num_opps_left > 0: # if enemy agents still alive and all coins have been collected
        return 2 # terminator mode
    else:
        if visible_coins > 0:  # if no. visible coins is greater or equal to 1
            return 0  # coin collecting mode
        elif coins_remaining > 0:  # if no. visible coins is 0
            return 1  # bombing/crate destroying mode
        else:
            return 1
            #print(f"Visible Coins: {visible_coins}. Coins remaining: {coins_remaining}. Opponents left: {num_opps_left}")


def store_unrecoverable_infos_helper(old_game_state, new_game_state):
    """
    Returns number of in this round collected coins, enemies that just died with their points
    """

    coins_old = set(old_game_state["coins"])
    coins_new = set(new_game_state["coins"])

    opponents_old = {other[0] for other in old_game_state["others"]}
    opponents_new = {other[0] for other in new_game_state["others"]}
    opponents_killed_names = opponents_old.difference(opponents_new)
    opponents_scores = {}

    for opp in old_game_state["others"]:
        if opp[0] in opponents_killed_names:
            opponents_scores[opp[0]] = opp[1]

    return len(coins_old.difference(coins_new)), opponents_scores


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
        neighboring_tiles = get_neighbor_coords(origin_xy) # [up, right, down, left]

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
        max_blow_count = surrounding_blow_count(origin_xy, arena, enemy_locations)
        most_destructive_tile = origin_xy

        for coord in neighboring_tiles:
            blow_count = surrounding_blow_count(coord, arena, enemy_locations)
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

        # print(np.array([[0, self.features[0], 0],
        #                 [self.features[3], self.features[4], self.features[1]],
        #                 [0, self.features[2], self.features[5]]]))

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



class PreviousWinnerCD(Features):
    # features used by 'No Time For Caution' with certain death check
    __slots__ = "features"
    features : list[int]

    def __init__(self, game_state: dict):
        self.features = [0] * 6

        ### Relevant Game State Information
        arena = np.array([game_state["field"]])[0] # todo: whats that np.array & [0] for?
        coins = game_state["coins"]
        own_bomb_pos = game_state["own_bomb"]
        bombs = []
        bombs_countdown = []
        if len(game_state["bombs"]):
            for i in range(len(game_state["bombs"])):
                bombs.append(game_state["bombs"][i][0])
                bombs_countdown.append(game_state["bombs"][i][1])
        blast_coords = []
        enemy_blast_coords = []
        blast_countdowns = []
        blast_to_bomb = []
        for idx, bomb_coord in enumerate(bombs):
            for coord in get_blast_coords(bomb_coord, arena):
                if coord not in blast_coords:
                    blast_coords.append(coord)
                    blast_countdowns.append(bombs_countdown[idx])
                    if own_bomb_pos is not None:
                        if bomb_coord != own_bomb_pos:
                            enemy_blast_coords.append(coord)
                    blast_to_bomb.append(idx)
                else:
                    idxx = blast_coords.index(coord)
                    blast_countdowns[idxx] = min(blast_countdowns[idxx], bombs_countdown[idx])
                    if blast_countdowns[idxx] > bombs_countdown[idx]:
                        blast_to_bomb[idxx] = idx
        explosions = game_state["explosion_map"]

        # player agent location
        player_pos = game_state["self"][3]

        # neighbouring tiles
        neighboring_tiles = get_neighbor_coords(player_pos) # [up, right, down, left]

        # enemy locations
        enemy_locations = []
        if len(game_state["others"]):
            for i in range(len(game_state["others"])):
                enemy_locations.append(game_state["others"][i][3])


        ### Fill up features
        game_mode = get_mode(len(game_state["coins"]), game_state["remaining_coins"], len(game_state["others"]))  # game mode feature

        # Calculate deadly coords
        deadly_coords = update_deadly(blast_coords, blast_countdowns, explosions)

        # find which neighboring tile, if any, yields certain death
        yields_certain_death = [False] * 4
        for idx, coord in enumerate(neighboring_tiles):
            if coord in deadly_coords:
                yields_certain_death[idx] = True


        # information on neighboring fields
        self.features[:4] =\
            get_neighboring_tile_infos(player_pos, neighboring_tiles, coins, bombs, arena, enemy_locations, game_mode,
                                          enemy_blast_coords, yields_certain_death, blast_coords, blast_countdowns, blast_to_bomb)
        # information on current field
        can_bomb = safe_to_bomb(player_pos, arena, game_state["self"][2], deadly_coords)

        self.features[4] = get_current_tile_info(player_pos, can_bomb, bombs, deadly_coords, blast_coords, blast_countdowns, explosions, game_mode, arena, enemy_locations)  # tile self

        # game mode
        self.features[5] = game_mode


    def __repr__(self):
        return ''.join(str(e) for e in self.features)


    def get_all_sym_str(self) -> tuple[list[str], list[tuple[int, int]]]:
        feature_as_matrix = np.array([[self.features[0], self.features[1]],
                                      [self.features[3], self.features[2]]])
        transforms = []
        sym_str = []

        for rot in range(4):
            for flip in [0, 1]:
                transforms.append((rot, flip))

                transformed_matrix = np.rot90(feature_as_matrix, k=rot)
                if flip:
                    transformed_matrix = np.flipud(transformed_matrix)
                sym_str.append(str(transformed_matrix[0,0]) + str(transformed_matrix[0,1]) + str(transformed_matrix[1,1])
                               + str(transformed_matrix[1,0]) + ''.join(str(e) for e in self.features[4:]))

        return sym_str, transforms

    def print_me(self):
        print(np.array([[0, self.features[0], 0],
                        [self.features[3], self.features[4], self.features[1]],
                        [0, self.features[2], self.features[5]]]))
        pass


