import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, model_evaluation, ACTIONS, rule_based

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    self.transitions = []
    pass

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if not old_game_state:
        return
    
    features_old = state_to_features(old_game_state)
    
    outputs = ['NONE', 'CURRENT', 'DOWN', 'UP', 'RIGHT', 'LEFT']
    safe_direction = outputs[features_old[0]]
    coin_direction = outputs[features_old[1]]
    crate_direction = outputs[features_old[2]]
    safe_to_bomb = features_old[3]
    enemy_direction = features_old[4]
    enemy_trapped = features_old[5]
    bomb_available = features_old[6]

    if safe_direction != self_action and safe_direction in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        events.append('IGNORED_THREAT')

    if coin_direction == self_action:
        events.append('MOVED_TOWARDS_COIN')

    if crate_direction == self_action:
        events.append('MOVED_TOWARDS_CRATE')

    if crate_direction != 'CURRENT' and enemy_trapped != 0 and self_action == 'BOMB':
        events.append('USELESS_BOMB') 

    _, _, _, agent_pos = old_game_state['self']
    field = old_game_state['field']
    f1 = field[agent_pos[0]-1, agent_pos[1]] == 1
    f2 = field[agent_pos[0]+1, agent_pos[1]] == 1
    f3 = field[agent_pos[0], agent_pos[1]-1] == 1
    f4 = field[agent_pos[0], agent_pos[1]+1] == 1
    if self_action == 'BOMB' and (f1 or f2 or f3 or f4):
        events.append('PLANTED_BOMB_NEXT_TO_CRATE')
    
    if safe_to_bomb == 0 and self_action == 'BOMB':
        events.append('BAD_BOMB')

    self.transitions.append(Transition(features_old, 
        self_action, 
        state_to_features(new_game_state), 
        reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), 
        last_action, 
        None, 
        reward_from_events(self, events)))

    q_update_transitions(self)
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    '''model = np.argmax(self.model, axis=-1)
    correct = 0
    incorrect = 0
    for features in np.ndindex(model.shape):
        if features[5] == 1 or features[4] != 0:    continue
        if rule_based(features) == ACTIONS[model[features]]:
            correct += 1
        else:
            incorrect += 1
    f = open('improvement.txt', 'a')
    f.write(f'{correct}:{incorrect},')
    f.close()

    _, s, _, _ = last_game_state['self']
    f = open('score_list.txt', 'a')
    f.write(str(s)+',')
    f.close()'''


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.MOVED_LEFT: -.01,
        e.MOVED_RIGHT: -.01,
        e.MOVED_UP: -.01,
        e.MOVED_DOWN: -.01,
        e.WAITED: -.01,
        e.INVALID_ACTION: -10, 
        'IGNORED_THREAT': -5,
        'PLANTED_BOMB_NEXT_TO_CRATE': 1,
        'MOVED_TOWARDS_COIN': 2,
        'MOVED_TOWARDS_CRATE': .5,
        'BAD_BOMB': -10,
        'USELESS_BOMB': -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# q-learing Update
def q_update_transitions(self):
    alpha = .2
    gamma = .8
    
    while self.transitions:
        old, action, new, reward = self.transitions.pop()
        idx_action = ACTIONS.index(action) if action else 4

        old.append(idx_action)

        if new:
            lala = alpha * (reward + gamma * np.max(
                model_evaluation(self.model, new)) - model_evaluation(self.model, old))
            self.model[
                old[0], 
                old[1], 
                old[2], 
                old[3],
                old[4],
                old[5],
                old[6],  
                idx_action
            ] += lala
        else:
            lala = alpha * (reward - model_evaluation(self.model, old))
            self.model[
                old[0], 
                old[1], 
                old[2], 
                old[3], 
                old[4],
                old[5],
                old[6], 
                idx_action
            ] += lala