from typing import List
import json
import numpy as np

from .callbacks import ALGORITHM
from .callbacks import check_state_exist_and_add, check_state_exist_w_sym, action_layout_to_action, act
from agent_code.utils.features import state_dict_to_feature_str, store_unrecoverable_infos_helper
from agent_code.utils.reward_sets import RewardGiver
from agent_code.utils.custom_events import state_to_events
from agent_code.utils.parameter_decay import AlphaDecayer


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Instantiate Eligibility Trace (dict of array with array[0] the action, array[1] the trace)
    if self.use_lambda:
        self.eligibility_trace = {}

    # Setup rewards
    self.learner = AlphaDecayer(self.learning_type, self.learning_param)
    self.reward_giver = RewardGiver(self.event_reward_set, self.dyn_rewards)

    self.can_act = False
    self.next_action = None


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
        self.can_act = True
        self.next_action = self.ACTIONS.index(act(self, new_game_state))
        self.can_act = False
        return

    ### Store unrecoverable game state information
    if self.feature in ["PreviousWinner", "DeusExMachinaFeatures"]:
        # Old game state of training is (new) game_state of act
        old_game_state["remaining_coins"] = self.remaining_coins_new
        old_game_state["own_bomb"] = self.own_bomb_new

        coin_diff, _ = store_unrecoverable_infos_helper(old_game_state, new_game_state)

        remaining_coins_new = self.remaining_coins_new - coin_diff
        own_bomb_new = self.own_bomb_new
        if own_bomb_new is None:
            if new_game_state["self"][3] in [bomb[0] for bomb in new_game_state["bombs"]]:
                own_bomb_new = new_game_state["self"][3]
        else:
            if own_bomb_new not in [bomb[0] for bomb in new_game_state["bombs"]]:
                own_bomb_new = None

        new_game_state["remaining_coins"] = remaining_coins_new
        new_game_state["own_bomb"] = own_bomb_new

    ### Calculate custom events from states
    events.extend(state_to_events(old_game_state, self_action, new_game_state, self.killed_opponents_scores, False))
    if not self.train_fast:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
        print(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    ### Transform state to all equivalent strings and action to index
    debug_old_state_str = old_state_str = self.prev_game_state_str
    #debug_old_state_str = old_state_str = state_dict_to_feature_str(old_game_state, self.feature)
    new_state_str = state_dict_to_feature_str(new_game_state, self.feature)

    action = self.ACTIONS.index(self_action)


    # Symmetry check
    new_state_transform = None
    if type(old_state_str) is not str:
        if self.feature not in ["RollingWindow", "DeusExMachinaFeatures"]:
            self.logger.warn(f"Non-single-state-string not implemented for {self.feature} yet.")

        # old state
        old_idx = check_state_exist_w_sym(self, old_state_str[0])
        if old_idx is None or old_state_str[1][0] is None:
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

        # new state
        new_idx = check_state_exist_w_sym(self, new_state_str[0])
        if new_idx is None:
            # Just use the first entry w/o transform
            new_state_str = new_state_str[0][0]

            self.next_game_state_str = ([new_state_str], [new_state_transform])

            self.can_act = True
            self.next_action = next_action = self.ACTIONS.index(act(self, new_game_state))
            self.can_act = False

        else:
            # Use transformed
            new_state_transform = new_state_str[1][new_idx]
            new_state_str = new_state_str[0][new_idx]

            self.next_game_state_str = ([new_state_str], [new_state_transform])

            self.can_act = True
            self.next_action = next_action = self.ACTIONS.index(act(self, new_game_state))
            self.can_act = False

            if not (next_action >= 4): # wait or bomb
                transformed_action_layout = np.rot90(self.action_layout, k=new_state_transform[0])
                if new_state_transform[1]:
                    transformed_action_layout = transformed_action_layout.T
                next_action = action_layout_to_action(self, transformed_action_layout, next_action)
    else:
        self.next_game_state_str = new_state_str
        self.can_act = True
        self.next_action = next_action = self.ACTIONS.index(act(self, new_game_state))
        self.can_act = False

    #check_state_exist_and_add(self, new_state_str) # todo: should be unnecessary since act is called with new state

    # old state-action pair visited ++1
    try:
        self.n_table[old_state_str][action] += 1
    except KeyError:
        print({old_state_str not in self.n_table.keys()})
        print({old_state_str not in self.q_table.keys()})
        print(old_game_state)
        print(self.prev_game_state)
        print("------------------------")
        print(old_state_str)
        print(debug_old_state_str)
        print(state_dict_to_feature_str(self.prev_game_state, self.feature))
        raise

    ### Calculate rewards
    reward = self.reward_giver.rewards_from_events(events) +\
             self.reward_giver.dynamic_rewards(old_game_state, self_action, new_game_state)

    if not self.train_fast:
        print(f"Reward {reward}")


    ### Update Q-Value
    q_old = self.q_table[old_state_str][action]
    q_update = reward + self.gamma * self.q_table[new_state_str][next_action]
    error = q_update - q_old

    # increase trace amount for visited state-action pair
    if self.use_lambda:
        self.eligibility_trace[old_state_str] = [action, 1]

    # Q update and decay eligibility trace after update
    if self.use_lambda:
        for state_str, trace in self.eligibility_trace.items():
            self.q_table[state_str][trace[0]] += self.learner.alpha(self.n_table[state_str][trace[0]]) * error * trace[1]
            trace[1] *= self.gamma * self.lambda_
    else:
        self.q_table[old_state_str][action] += self.learner.alpha(self.n_table[old_state_str][action]) * error

    ### Save state string and transform in short term memory
    self.total_rewards += reward


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



    ### Store unrecoverable game state information
    if self.feature in ["PreviousWinner", "DeusExMachinaFeatures"]:
        # Old game state of training is (new) game_state of act
        self.prev_game_state["remaining_coins"] = self.remaining_coins_new
        self.prev_game_state["own_bomb"] = self.own_bomb_new
        coin_diff, _ = store_unrecoverable_infos_helper(self.prev_game_state, last_game_state)

        remaining_coins_new = self.remaining_coins_new - coin_diff
        own_bomb_new = self.own_bomb_new
        if own_bomb_new is None:
            if last_game_state["self"][3] in [bomb[0] for bomb in last_game_state["bombs"]]:
                own_bomb_new = last_game_state["self"][3]
        else:
            if own_bomb_new not in [bomb[0] for bomb in last_game_state["bombs"]]:
                own_bomb_new = None

        last_game_state["remaining_coins"] = remaining_coins_new
        last_game_state["own_bomb"] = own_bomb_new

    ### Calculate custom events from states
    events.extend(
        state_to_events(self.prev_game_state, last_action, last_game_state, self.killed_opponents_scores, False))

    if not self.train_fast:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {last_game_state["step"]}')
        print(f'Encountered game event(s) {", ".join(map(repr, events))} in step {last_game_state["step"]}')

    ### Transform state to string and action to index
    old_state_str = self.prev_game_state_str
    #old_state_str = state_dict_to_feature_str(self.prev_game_state, self.feature)
    action = self.ACTIONS.index(last_action)

    # Symmetry check
    if type(old_state_str) is not str:
        if self.feature not in ["RollingWindow", "DeusExMachinaFeatures"]:
            self.logger.warn(f"Non-single-state-string not implemented for {self.feature} yet.")
        old_idx = check_state_exist_w_sym(self, old_state_str[0])
        if old_idx is None or old_state_str[1][0] is None:
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

    # old state-action pair visited ++1
    #check_state_exist_and_add(self, old_state_str)
    try:
        self.n_table[old_state_str][action] += 1
    except KeyError:
        print("\nNo")
        print({old_state_str not in self.n_table.keys()})
        print({old_state_str not in self.q_table.keys()})
        print(old_state_str)
        print(state_dict_to_feature_str(self.prev_game_state, self.feature))
        raise

    ### Update Q-Value
    q_old = self.q_table[old_state_str][action]
    q_update = self.reward_giver.rewards_from_events(events) +\
             self.reward_giver.dynamic_rewards(self.prev_game_state, last_action, last_game_state)
    error = q_update - q_old

    # increase trace amount for visited state-action pair
    if self.use_lambda:
        self.eligibility_trace[old_state_str] = [action, 1]

    # Q update
    if self.use_lambda:
        for state_str, trace in self.eligibility_trace.items():
            self.q_table[state_str][trace[0]] += self.learner.alpha(self.n_table[state_str][trace[0]]) * error * trace[1]
    else:
        self.q_table[old_state_str][action] += self.learner.alpha(self.n_table[old_state_str][action]) * error

    if not self.train_fast:
        print(f"Reward {q_update}")

    # Reset eligibility trace
    if self.use_lambda:
        self.eligibility_trace = {}

    # Store the q_table as json every n rounds
    if (last_game_state["round"] % self.save_n_rounds) == 0:
        with open(self.q_table_filename, "w") as q_table_file, open(self.n_table_filename, "w") as n_table_file:
            q_table = self.q_table
            n_table = self.n_table
            q_table["meta"] = {"algorithm": ALGORITHM, "feature": self.feature, "q_table_id": self.q_table_id}
            n_table["meta"] = {"algorithm": ALGORITHM, "feature": self.feature, "q_table_id": self.q_table_id}
            json.dump(q_table, q_table_file, indent=4, sort_keys=True)
            json.dump(n_table, n_table_file, indent=4, sort_keys=True)

        with open("rewards.txt", "a") as rewards_file:
            rewards_file.write(str(self.total_rewards / last_game_state["round"]) + "\n")

