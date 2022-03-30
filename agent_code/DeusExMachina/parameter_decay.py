import math
import random
import importlib
from types import SimpleNamespace



# Explorer Class
class Explorer:
    __slots__ = ("exp_str", "exp_param", "piggy")
    exp_str : str
    exp_param : float

    def __init__(self, exp_str: str, param = None) -> None:
        self.exp_str = exp_str
        if param is not None:
            self.exp_param = param

    def explore(self, available_actions, q_values, counts):
        return explore_function_dict[self.exp_str](self.exp_param, available_actions, q_values, counts)



# Piggyback Class
class PiggyCarry:
    #__slots__ = ("piggy", "piggy_name", "epsilon", "callbacks", "agent")
    piggy_name : str
    epsilon : float

    def __init__(self, logger, agent_name, epsilon):
        self.piggy_name = agent_name
        self.epsilon = epsilon

        self.piggy = SimpleNamespace()
        self.piggy.train = False
        self.piggy.logger = logger

        self.callbacks = importlib.import_module('agent_code.' + agent_name + '.callbacks')
        self.callbacks.setup(self.piggy)



    def carry(self, game_state: dict):
        if random.random() > self.epsilon:
            # some actions may have the same value, randomly choose one of these actions
            action = self.callbacks.act(self.piggy, game_state)
        else:
            # choose random action
            action = random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])

        return action



# Exploration Update Function
class QUpdater:
    __slots__ = ("upd_str", "upd_param", "piggy")
    upd_str: str
    upd_param: float

    def __init__(self, upd_str: str, param=None) -> None:
        self.upd_str = upd_str
        if param is not None:
            self.upd_param = param

    def update(self, q_values, counts):
        return update_function_dict[self.upd_str](self.upd_param, q_values, counts)


# Alpha Decayer Class
class AlphaDecayer:
    __slots__ = ("decay_str", "decay_param")
    decay_str : str
    decay_param : float

    def __init__(self, decay_str: str, param = None) -> None:
        self.decay_str = decay_str
        if param is not None:
            self.decay_param = param

    def alpha(self, visit_count):
        return alpha_decay_function_dict[self.decay_str](visit_count, self.decay_param)



### Explorer Functions

def best(factor, available_actions, q_values, counts):
    return random.choice([idx for idx, val in enumerate(q_values) if val == max(q_values)])

def epsilon_greedy(epsilon, available_actions, q_value, counts):
    if random.random() > epsilon:
        # some actions may have the same value, randomly choose one of these actions
        action = random.choice([idx for idx, val in enumerate(q_value) if val == max(q_value)])
    else:
        # choose random action
        action = random.choice(available_actions)

    return action

def epsilon_decay(epsilon, available_actions, q_value, counts):
    eps = 1/(sum(counts) - len(available_actions) + 1)
    if random.random() > eps:
        # some actions may have the same value, randomly choose one of these actions
        action = random.choice([idx for idx, val in enumerate(q_value) if val == max(q_value)])
    else:
        # choose random action
        action = random.choice(available_actions)

    return action

def explore_function_standard(factor, available_actions, q_value, counts):
    values = []
    for i in range(6):
        values.append(q_value[i] + factor / counts[i])
    return random.choice([idx for idx, val in enumerate(values) if val == max(values)])



explore_function_dict = {"best": best,
                         "e_greedy": epsilon_greedy,
                         "e_decay": epsilon_decay,
                         "explore_standard": explore_function_standard}



### Update Functions

def q_max(factor, q_values, counts):
    return max(q_values)

def k_over_n(factor, q_values, counts):
    values = []
    for i in range(6):
        values.append(q_values[i] + factor / counts[i])
    return max(values)

# Update Dictonary

update_function_dict = {
    "q_max": q_max,
    "k_over_n": k_over_n
}



### Alpha Decay Functions

def alpha_const(visit_count: int, value):
    return value

def alpha_over_n(visit_count: int, value):
    return value / visit_count

def alpha_ln(visit_count: int, value):
    return value * math.log(visit_count + 1) / visit_count

# Alpha Decay Dictionary
alpha_decay_function_dict = {"const": alpha_const,
                             "1/n": alpha_over_n,
                             "ln/n": alpha_ln}

