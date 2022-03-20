import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Own events
    old_f = state_to_features(old_game_state)
    if old_f is not None:
        action_num = np.argmax(self_action == np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])[:])
        if old_f[1] != 5 and old_f[1] != action_num:
            events.append('NOT_DOGED') # ignored doge direction while in danger
        if old_f[1] != 5 and old_f[6] == action_num:
            events.append('MOVED_TO_TARGET')
        if old_f[1] != 5 and old_f[0] == 1 and action_num == 5:
            events.append('GOOD_BOMB_DROP')
            
    if len(self.transitions) == 4:
        q_update_transitions(self)
        
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    if old_game_state is None:
        self.logger.debug(f'Encountered game type problem in step {new_game_state["step"]}')
        self.transitions.pop() #the appended element is invalide for q_update


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    q_update_transitions(self)
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.MOVED_LEFT: -.01,
        e.MOVED_RIGHT: -.01,
        e.MOVED_UP: -.01,
        e.MOVED_DOWN: -.01,
        e.WAITED: -.01,
        e.INVALID_ACTION: -1, 
        e.BOMB_DROPPED: -1,
        'GOOD_BOMB_DROP': 1.1,
        'MOVED_TO_TARGET': 0.01,
        'NOT_DODGED': -1
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# q-learing Update
def q_update_transitions(self):
    alpha = 0.1 # learning rate
    gamma = 0.1 #importace of future
    elem = len(self.transitions)
    
    for i in range(elem):
            old, a, new, reward = self.transitions.pop()
            action_num = np.argmax(a == np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])[:])
            if new is None:
                self.model[tuple(np.concatenate((np.zeros(1), old, np.array([action_num]))).astype(int))] = (1 - alpha) * self.model[tuple(np.concatenate((np.zeros(1), old, np.array([action_num]))).astype(int))] + alpha * reward
                #increment count of (state, action) pair
                self.model[tuple(np.concatenate((np.ones(1), old, np.array([action_num]))).astype(int))] += 1
            else: 
                #update entry in table
                self.model[tuple(np.concatenate((np.zeros(1), old, np.array([action_num]))).astype(int))] = (1 - alpha) * self.model[tuple(np.concatenate((np.zeros(1), old, np.array([action_num]))).astype(int))] + alpha * (reward + gamma * np.max(self.model[tuple(np.concatenate((np.zeros(1), new)).astype(int))]))
                #increment count of (state, action) pair
                self.model[tuple(np.concatenate((np.ones(1), old, np.array([action_num]))).astype(int))] += 1