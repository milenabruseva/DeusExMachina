from typing import List
import json

import events as e
import settings as s
from ..custom_events import reward_from_events
from ..custom_events import CustomEvents as ce
from .callbacks import check_state_exist
from ..features import LocalVision


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    pass


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

    if not self.train_fast:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state is None:
        return

    old_state_str = str(LocalVision(old_game_state))
    new_state_str = str(LocalVision(new_game_state))

    # Calculate custom events from states
    events.extend(state_to_events(self, old_game_state, self_action, new_game_state))
    reward = reward_from_events(events)

    check_state_exist(self,new_state_str)
    q_old = self.q_table.loc[old_state_str, self_action]

    q_update = reward + self.gamma * self.q_table.loc[new_state_str, :].max()
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
    if not self.train_fast:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {last_game_state["step"]}')

    old_state_str = str(LocalVision(last_game_state))
    check_state_exist(self, old_state_str)
    q_old = self.q_table.loc[old_state_str, last_action]

    # Calculate custom events from states
    events.extend(state_to_events(self, None, last_action, last_game_state))
    q_update = reward_from_events(events)
    self.q_table.loc[old_state_str, last_action] += self.lr * (q_update - q_old)

    # Store the q_table as json
    with open("q_table.json", "w") as q_table_file:
        json.dump(self.q_table.to_json(orient="index"), q_table_file, ensure_ascii=False, indent=4)


def state_to_events(self, old_game_state: dict, action_taken: str, new_game_state: dict) -> List[str]:
    custom_events = []

    if old_game_state is None:
        # End of round
        won = True
        my_score = new_game_state["self"][1]
        tot_enemy_score = sum([enemy[1] for enemy in new_game_state["others"]])
        points_left = (3*5 + 9*1) - (my_score + tot_enemy_score)

        # Check if probably won
        for enemy in new_game_state["others"]:
            if my_score < enemy[1] + points_left:
                won = False
                break

        if won:
            custom_events.append(ce.PROBABLY_WON)
        else:
            custom_events.append(ce.PROBABLY_LOST)

    else:
        pass

    return custom_events