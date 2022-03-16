from abc import ABCMeta, abstractmethod
import numpy as np


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


def get_tile_type(coord, coins, bombs, arena):
    if coord in coins:
        return 2
    elif coord in bombs:
        return -1
    else:
        return arena[coord]


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
