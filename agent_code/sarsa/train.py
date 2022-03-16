from typing import List
import json

import events as e
from .callbacks import state_to_features, check_state_exist, act


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    with open("rewards.json") as rewards_file:
        params = json.load(rewards_file)
        self.rewards = {
            "won": params["won"],
            "lost": params["lost"]
        }


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state is None:
        return

    old_state_str = str(state_to_features(old_game_state))
    new_state_str = str(state_to_features(new_game_state))
    reward = reward_from_state(self, new_game_state)
    next_action = act(self, new_game_state)

    check_state_exist(self,new_state_str)
    q_old = self.q_table.loc[old_state_str, self_action]

    q_update = reward + self.gamma * self.q_table.loc[new_state_str, next_action]
    self.q_table.loc[old_state_str, self_action] += self.lr * (q_update - q_old)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    old_state_str = str(state_to_features(last_game_state))
    check_state_exist(self, old_state_str)
    q_old = self.q_table.loc[old_state_str, last_action]
    won = True
    my_score = last_game_state["self"][1]

    for enemy in last_game_state["others"]:
        if my_score < enemy[1]:
            won = False
            break
    reward = self.rewards["won"] if won else self.rewards["lost"]

    q_update = reward
    self.q_table.loc[old_state_str, last_action] += self.lr * (q_update - q_old)

    # Store the q_table as json
    with open("q_table.json", "w") as q_table_file:
        json.dump(self.q_table.to_json(orient="index"), q_table_file, ensure_ascii=False, indent=4)


def reward_from_state(self, game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }

    # Calculate reward
    reward = 0

    return reward
