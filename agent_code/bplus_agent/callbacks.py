import os
import pickle
import random
import numpy as np
from sklearn import tree

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
min_count = 3 #required count to be included in regression


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
#   if self.train or not os.path.isfile("my-saved-model.pt"):
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
      #  weights = np.random.rand(len(ACTIONS))
       # self.model = weights / weights.sum()
        self.model = np.zeros((2, 2, 6, 2, 2, 2, 2, 4, 7, 4, 6)) # 1-9 dim featurespace, 10 Actions, 0 switch model/count
        
    elif self.train:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.clf = build_regression(self)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    epsilon = 0.2 
    tol = 0.2 #if action1 - action2 <tol => regression
    trust_mark = 5 #minimal data on action to trust
    switch_detfunc = 1 # 0 to switch to regular training
    
    if self.train and random.random() < epsilon:  #Max-Boltzmann exploration strategy
        features = state_to_features(game_state)
        T = 1 # heat parameter, not used
        prob = np.exp(self.model[tuple(np.concatenate((np.zeros(1), features)).astype(int))]/T)
        prob = prob / np.sum(prob)
        return np.random.choice(ACTIONS, p=prob)
    
    if self.train and random.random() < switch_detfunc:
        return best_action(self, state_to_features(game_state))
    
    if hasattr(self, 'model'):
        features = state_to_features(game_state)
        if features is None:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        action = np.argmax(self.model[tuple(np.concatenate((np.zeros(1), features)).astype(int))])
        count = self.model[tuple(np.concatenate((np.ones(1), features, np.array([action]))).astype(int))]
        action2 = np.argmax(self.model[tuple(np.concatenate((np.zeros(1), features)).astype(int))] - np.eye(6)[action] * tol) 
        if action != action2 or count < trust_mark:
            return regression(features, self.clf)
        return ACTIONS[action]
    self.logger.debug('No model')
    return 'WAIT'


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of the model.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    _, _, bomb_ready, a_pos = game_state['self']
    field = game_state['field'] + np.floor(game_state['explosion_map']* 0.5)* 5  # set explosions with timer 2 to 5, exp with timer 1 are irrelevant
    
    # Set the other agents
    direction_opponent = 5 #default for no opponent in bomb range
    for foe in game_state['others']:
        _, _, _, pos = foe
        x, y = pos
        field[x, y] = 3
        if x == a_pos[0] and abs(y - a_pos[1]) <= 3:
            direction_opponent = 4
        if y == a_pos[1] and abs(x- a_pos[0]) <= 3:
            direction_opponent = 4
        
    # Set the visable coins
    for x, y in game_state['coins']:
        field[x, y] = 2
    
    #Set bomb=4 und 5 on tiles in reach of a bomb with timer 0, ..., 8 of bomb in reach with timer 3
    for bomb in game_state['bombs']:
        pos, t = bomb 
        x, y = pos
        field[x, y] = 4
        free_direction = np.array([1, 1, 1, 1])
        for i in range(1, 4): # Set the tiles in range of expolsion
            if free_direction[0] == 1:
                if field[x-i, y] != -1:
                    if field[x-i, y] == 0 or field[x-i, y]> t + 4:
                        field[x-i, y] = t + 5
                else:
                    free_direction[0] = 0
            if free_direction[1] == 1:
                if field[x+i, y] != -1:
                    if field[x+i, y] == 0 or field[x+i, y]> t + 4:
                        field[x+i, y] = t + 5
                else:
                    free_direction[1] = 0
            if free_direction[2] == 1:
                if field[x, i+y] != -1:
                    if field[x, i+y] == 0 or field[x, i+y]> t+ 4:
                        field[x, i+y] = t + 5
                else:
                    free_direction[2] = 0
            if free_direction[3] == 1:
                if field[x, y-i] != -1:
                    if field[x, y-i] == 0 or field[x, y-i]> t+ 4:
                        field[x, y-i] = t + 5
                else:
                    free_direction[3] = 0
    
    #search coin
    if len(game_state['coins'])>0:
        board = np.minimum(field, 0) - (field == 1) - (field == 3) - (field == 4)
        board[a_pos[0], a_pos[1]] = - 1
        paths = []
        direction_coin = 5 # default 5 no coin reachable
        # first step outside of loop to set direction of the first step of the path
        # 0-3 correspond to move direction in ACTION
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 1)) #1 path starts RIGHT
            board[a_pos[0] + 1, a_pos[1]] = -1 # so no tile is part of two paths
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 3)) #3 path starts LEFT
            board[a_pos[0] - 1, a_pos[1]] = -1
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0], 1 + a_pos[1]] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 2)) #2 path starts DOWN
            board[a_pos[0], a_pos[1]+ 1] = -1
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 0)) #0 path starts UP
            board[a_pos[0], a_pos[1]- 1] = -1
        while len(paths)>0 and direction_coin == 5: #breath searche
            x, y , z = paths.pop(0)
            if field[x, y] == 2:
                direction_coin = z
            if board[x + 1, y] == 0:
                paths.append((x + 1, y, z))
                board[x + 1, y] = -1
            if board[x - 1, y] == 0:
                paths.append((x - 1, y, z))
                board[x - 1, y] = -1
            if board[x, y + 1] == 0:
                paths.append((x, y + 1, z))
                board[x, y + 1] = -1
            if board[x, y - 1] == 0:
                paths.append((x, y - 1, z))
                board[x, y - 1] = -1
    else:
        direction_coin = 8 # no visable coins on the board
    #search next crate if no coin reachable
    if direction_coin == 8 or direction_coin == 5:
        #same as for coins just with crate treated as walkable
        board = np.minimum(field, 0)  - (field == 3) - (field == 4)
        board[a_pos[0], a_pos[1]] = - 1
        paths = []
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 5)) #1 RIGHT
            board[a_pos[0] + 1, a_pos[1]] = -1 
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 7)) #3 LEFT
            board[a_pos[0] - 1, a_pos[1]] = -1
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0], a_pos[1]+ 1] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 6)) #2 DOWN
            board[a_pos[0], a_pos[1]+ 1] = -1
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 4)) #0 UP
            board[a_pos[0], a_pos[1]- 1] = -1
        while len(paths)>0 and direction_coin == 8: #breath search
            x, y , z = paths.pop(0)
            if field[x, y] == 1:
                direction_coin = z
            if board[x + 1, y] == 0:
                paths.append((x + 1, y, z))
                board[x + 1, y] = -1
            if board[x - 1, y] == 0:
                paths.append((x - 1, y, z))
                board[x - 1, y] = -1
            if board[x, y + 1] == 0:
                paths.append((x, y + 1, z))
                board[x, y + 1] = -1
            if board[x, y - 1] == 0:
                paths.append((x, y - 1, z))
                board[x, y - 1] = -1
    
    #search opponent
    # again the same as coin search just different targets and agent count as walkable
    if len(game_state['others'])>0 and direction_opponent == 5: #not nessecary of no opponets exist or one is already in bomb range (direction_opponent ==4)
        board = np.minimum(field, 0) - (field == 1) - (field == 4)
        board[a_pos[0], a_pos[1]] = - 1
        paths = []
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 1))
            board[a_pos[0] + 1, a_pos[1]] = -1
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 3))
            board[a_pos[0] - 1, a_pos[1]] = -1
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0], a_pos[1]+ 1] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 2))
            board[a_pos[0], a_pos[1]+ 1] = -1
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 0))
            board[a_pos[0], a_pos[1]- 1] = -1
        while len(paths)>0 and direction_opponent == 5:
            x, y , z = paths.pop(0)
            if field[x, y] == 3:
                direction_opponent = z
            if board[x + 1, y] == 0:
                paths.append((x + 1, y, z))
                board[x + 1, y] = -1
            if board[x - 1, y] == 0:
                paths.append((x - 1, y, z))
                board[x - 1, y] = -1
            if board[x, y + 1] == 0:
                paths.append((x, y + 1, z))
                board[x, y + 1] = -1
            if board[x, y - 1] == 0:
                paths.append((x, y - 1, z))
                board[x, y - 1] = -1
    
    #doge explosion
    doge_dir = find_doge_direction(field, a_pos) # 0-3 direction to dodge if in danger, 4 death unavoidable, 5 no danger
    
    #is a bomb drop safe?
    if bomb_ready == 1: #1=bomb ready
        field_b = np.copy(field)
        x, y = a_pos
        t = 4
        field_b[x, y] = 4 # simulate drop bomb at current position
        field_b[field_b >= 6] -= 1
        free_direction = np.array([1, 1, 1, 1])
        for i in range(1, 4): #same as set bomb above
            if free_direction[0] == 1:
                if field[x-i, y] != -1:
                    if field[x-i, y] == 0 or field[x-i, y]> t + 4:
                        field_b[x-i, y] = t + 4
                else:
                    free_direction[0] = 0
            if free_direction[1] == 1:
                if field[x+i, y] != -1:
                    if field[x+i, y] == 0 or field[x+i, y]> t + 4:
                        field_b[x+i, y] = t + 4
                else:
                    free_direction[1] = 0
            if free_direction[2] == 1:
                if field[x, i+y] != -1:
                    if field[x, i+y] == 0 or field[x, i+y]> t+ 4:
                        field_b[x, i+y] = t + 4
                else:
                    free_direction[2] = 0
            if free_direction[3] == 1:
                if field[x, y-i] != -1:
                    if field[x, y-i] == 0 or field[x, y-i]> t+ 4:
                        field_b[x, y-i] = t + 4
                else:
                    free_direction[3] = 0
        if find_doge_direction(field_b, a_pos) == 4: #4 we're dead
            bomb_ready = 2 #2 = bomb drop is SUICIDE
            
    #ring
    ring = max(abs(a_pos[0] - (field.shape[0]-1)/2), abs(a_pos[1] - (field.shape[1]-1)/2)) - 1
    
    #dense
    x_min = max(0, a_pos[0] - 2)
    x_max = min(field.shape[0] - 1, a_pos[0] + 2)
    y_min = max(0, a_pos[1] - 2)
    y_max = min(field.shape[1] - 1, a_pos[1] + 2)
    free_tils = np.sum(field[x_min:x_max, y_min:y_max] == 0)
    dense = round((20 - free_tils) / 7)
       
    # is a bomb drop possible and useful
    bomb_d = 0
    sumcrat = (field[a_pos[0]- 1, a_pos[1]] == 1) + (field[a_pos[0]+ 1, a_pos[1]] == 1) + (field[a_pos[0], a_pos[1]- 1] == 1) + (field[a_pos[0], a_pos[1]+ 1] == 1) 
    if int(bomb_ready) == 1 and sumcrat >= 1 and direction_coin>=4:
        bomb_d = 1
    if bomb_ready == 1 and direction_opponent == 4:
        bomb_d = 1
        
    # the actual features
    channels = []
    channels.append(bomb_d) #bomb 1, not 0
    channels.append(doge_dir)  #0-3 direction to doge exp, 4 undogeable, 5 no need to doge
    #the adjacent tiles to the current position
    reduce = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    # 0 for free(0),coin(2),exp in 2-4 steps(6-8); 1 for crate(1), opponent(3), exp in 1 step(5), walls(-1=9)
    channels.append(reduce[int(field[a_pos[0]- 1, a_pos[1]])]) #left
    channels.append(reduce[int(field[a_pos[0]+ 1, a_pos[1]])])# right
    channels.append(reduce[int(field[a_pos[0], a_pos[1]- 1])]) #up
    channels.append(reduce[int(field[a_pos[0], a_pos[1]+ 1])]) #down
    
    #direction_coin and direction_opponent are fused to driection_target
    direction_target = direction_coin%4
    if direction_coin == 8:
        direction_target = direction_opponent%4
    channels.append(direction_target)
    #0-3 direction of nearest opp, 4 opp in bomb range, 5 no opp reachabel/exist
    
    channels.append(ring)  #ring of the field outside(6) to inner(0)
    channels.append(dense)  #density freedom 0 free to 3 tight

    stacked_channels = np.stack(channels)
    stacked_channels.astype(int)
    # and return them as a vector
    
    #Overall feature  0     1     2    3     4     5     6     7     8
    #dimension        2     6     2    2     2     2     4     8     4
    #               bomb? doge_d  <-   ->   up   down target_d ring density
    return stacked_channels.reshape(-1).astype(int)

def find_doge_direction(field: np.array, a_pos: np.array) -> int:
    if field[a_pos[0], a_pos[1]] != 0:
        board = np.minimum(field, 0) - (field == 1) - (field == 3) - (field == 4)
        paths = []
        doge_dir = 4
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 1, 1))
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 3, 1))
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0],  1+ a_pos[1]] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 2, 1))
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 0, 1))
        time = 1
        while doge_dir == 4 and len(paths)>0:
            x, y , z, time = paths.pop(0)
            if field[x, y] - time <= 3 and time <=4:
                doge_dir = z
            if board[x + 1, y] == 0 and field[x + 1, y] - time != 5 and field[x + 1, y] - time != 4:
                paths.append((x + 1, y, z, time + 1))
            if board[x - 1, y] == 0 and field[x - 1, y] - time != 5 and field[x - 1, y] - time != 4:
                paths.append((x - 1, y, z, time + 1))
            if board[x, y + 1] == 0 and field[x, y + 1] - time != 5 and field[x, y + 1] - time != 4:
                paths.append((x, y + 1, z, time + 1))
            if board[x, y - 1] == 0 and field[x, y - 1] - time != 5 and field[x, y - 1] - time != 4:
                paths.append((x, y - 1, z, time + 1))
    else:
        doge_dir = 5
    return doge_dir

def best_action(self, features):
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    F2A = [0, 0, 3, 1, 0, 2]
    aim = [4, 3, 5, 2, 4, 3, 5, 2, 6][features[6]]
    if features[1] <= 3: #dog if in danger highest priority
        return ACTIONS[features[1]]
    if features[0] == 1:
        return ACTIONS[5]
    if features[6] <= 3 and features[aim] == 0:#go to t if one is reachable
        return ACTIONS[features[6]]
    return ACTIONS[4] 

def build_regression(self):
    dims = (self.model[0]).shape
    channel = []
    y = []
    #neither nice nor efficent but time is irrelevant in setup
    for a in range(dims[0]):
        for b in range(dims[1]):
            for c in range(dims[2]):
                for d in range(dims[3]):
                    for e in range(dims[4]):
                        for f in range(dims[5]):
                            for g in range(dims[6]):
                                for h in range(dims[7]):
                                    for i in range(dims[8]):
                                        action = np.argmax(self.model[0, a, b, c, d, e, f, g, h,i])
                                if self.model[1, a, b, c, d, e, f, g,h,i, action]  >= min_count:
                                    channel.append([a, b, c, d, e, f, g,h,i])
                                    y.append([action])
                                
    
    X = np.stack(channel)
    y = np.stack(y)
    clf = tree.DecisionTreeClassifier(min_samples_leaf = 4)
    clf = clf.fit(X, y)
    return clf

def regression(features, clf):
    return ACTIONS[clf.predict([features])[0]]
