from typing import List
import json
import numpy as np

import events as e
import settings as s

from .callbacks import ALGORITHM
from .callbacks import check_state_exist_and_add, check_state_exist_w_sym, action_layout_to_action, act
from ..features import state_dict_to_feature_str
from ..reward_sets import RewardGiver
from ..custom_events import state_to_events


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Setup rewards
    self.reward_giver = RewardGiver(self.event_reward_set, self.dyn_rewards)


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

    if old_game_state is None:
        return

    # Calculate custom events from states
    events.extend(state_to_events(old_game_state, self_action, new_game_state))
    if not self.train_fast:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Transform state to all equivalent strings and action to index
    old_state_str = state_dict_to_feature_str(old_game_state, self.feature)
    new_state_str = state_dict_to_feature_str(new_game_state, self.feature)

    action = self.ACTIONS.index(self_action)

    # Calculate rewards
    reward = self.reward_giver.rewards_from_events(events) +\
             self.reward_giver.dynamic_rewards(old_game_state, self_action, new_game_state)

    # Update Q-Value
    # Symmetry check
    if type(old_state_str) is not str:
        if self.feature != "RollingWindow":
            self.logger.warn("Non-single-state-string only implemented for RollingWindow yet.")
        old_idx = check_state_exist_w_sym(self, old_state_str[0])
        if old_idx is None:
            # Just use the first entry w/o transform
            old_state_str = old_state_str[0][0]
        else:
            # Use transformed
            transform = old_state_str[1][old_idx]
            old_state_str = old_state_str[0][old_idx]
            if not (action >= 4): # wait or bomb
                transformed_action_layout = np.rot90(self.action_layout, k=transform[0])
                if transform[1]:
                    transformed_action_layout = transformed_action_layout.T
                action = action_layout_to_action(self, transformed_action_layout, action)
        new_idx = check_state_exist_w_sym(self, new_state_str[0])
        if new_idx is None:
            # Just use the first entry w/o transform
            new_state_str = new_state_str[0][0]
        else:
            # Use transformed
            new_state_str = new_state_str[0][new_idx]

    check_state_exist_and_add(self, new_state_str)

    q_old = self.q_table[old_state_str][action]
    q_update = reward + self.gamma * max(self.q_table[new_state_str])
    self.q_table[old_state_str][action] += self.lr * (q_update - q_old)


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

    # Calculate custom events from states
    events.extend(state_to_events(None, last_action, last_game_state))
    if not self.train_fast:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {last_game_state["step"]}')

    # Transform state to string and action to index
    old_state_str = state_dict_to_feature_str(last_game_state, self.feature)
    action = self.ACTIONS.index(last_action)

    # Update Q-Value
    # Symmetry check
    if type(old_state_str) is not str:
        if self.feature != "RollingWindow":
            self.logger.warn("Non-single-state-string only implemented for RollingWindow yet.")
        old_idx = check_state_exist_w_sym(self, old_state_str[0])
        if old_idx is None:
            # Just use the first entry w/o transform
            old_state_str = old_state_str[0][0]
        else:
            # Use transformed
            transform = old_state_str[1][old_idx]
            old_state_str = old_state_str[0][old_idx]
            if not (action >= 4): # wait or bomb
                transformed_action_layout = np.rot90(self.action_layout, k=transform[0])
                if transform[1]:
                    transformed_action_layout = transformed_action_layout.T
                action = action_layout_to_action(self, transformed_action_layout, action)

    q_old = self.q_table[old_state_str][action]
    q_update = self.reward_giver.rewards_from_events(events) +\
             self.reward_giver.dynamic_rewards(last_game_state, last_action, None)
    self.q_table[old_state_str][action] += self.lr * (q_update - q_old)

    # Store the q_table as json every 100 rounds
    if (last_game_state["round"] % self.save_n_rounds) == 0:
        with open(self.proper_filename, "w") as q_table_file:
            q_table = self.q_table
            q_table["meta"] = {"algorithm": ALGORITHM, "feature": self.feature, "q_table_id": self.q_table_id}
            json.dump(q_table, q_table_file, indent=4, sort_keys=True)

