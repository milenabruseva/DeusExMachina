from typing import List
import events as e
from .custom_events import CustomEvents as ce
from .custom_events import manhattan_distance


### Event Reward Sets as dicts

game_rewards_no_1 = {
    e.COIN_COLLECTED: 10,
    e.COIN_FOUND: 3,
    e.KILLED_OPPONENT: 50,
    e.CRATE_DESTROYED: 3,
    e.GOT_KILLED: -50,
    e.KILLED_SELF: -50,
    e.INVALID_ACTION: -5,
    e.MOVED_UP: 0.1,
    e.MOVED_DOWN: 0.1,
    e.MOVED_RIGHT: 0.1,
    e.MOVED_LEFT: 0.1,
    e.WAITED: -0.1,
    e.BOMB_DROPPED: 2,
    ce.PROBABLY_WON: 100,
    ce.PROBABLY_LOST: -100
}

coin_heaven = {
    e.COIN_COLLECTED: 100,
    e.GOT_KILLED: -500,
    e.KILLED_SELF: -500,
    e.BOMB_DROPPED: -50,
    e.INVALID_ACTION: -50,
    e.WAITED: -50,
    ce.DECREASED_NEAREST_COIN_DIST: 10,
    ce.INCREASED_NEAREST_COIN_DIST: -20,
    ce.SAME_NEAREST_COIN_DIST: -1
}

coin_minimal = {
    e.COIN_COLLECTED: 100,
    #e.GOT_KILLED: -80,
    #e.KILLED_SELF: -50,
    e.INVALID_ACTION: -500,
    #ce.DECREASED_NEAREST_COIN_DIST: 10,
    ce.INCREASED_NEAREST_COIN_DIST: -11,
    ce.SAME_NEAREST_COIN_DIST: -11
}

classic = {
    e.COIN_COLLECTED: 10,
    e.GOT_KILLED: -50,
    e.KILLED_SELF: -50,
    e.INVALID_ACTION: -5,
    e.COIN_FOUND: 3,
    e.KILLED_OPPONENT: 50,
    e.CRATE_DESTROYED: 3,
    ce.DECREASED_NEAREST_COIN_DIST: 0.1,
    ce.INCREASED_NEAREST_COIN_DIST: -0.1,
    ce.IS_NOW_SAFE: 0.5,
    ce.IS_NOW_UNSAFE: -0.5,
    e.WAITED: -0.1,
    e.BOMB_DROPPED: 2,
    ce.DECREASED_NEAREST_ENEMY_DIST: 0.05,
    ce.INCREASED_NEAREST_ENEMY_DIST: -0.05
}

classic_plus = {
    e.COIN_COLLECTED: 100,
    e.GOT_KILLED: -50,
    e.KILLED_SELF: -30,
    e.INVALID_ACTION: -500,
    e.COIN_FOUND: 3,
    e.KILLED_OPPONENT: 500,
    e.CRATE_DESTROYED: 15,
    ce.DECREASED_NEAREST_COIN_DIST: 10,
    ce.INCREASED_NEAREST_COIN_DIST: -11,
    #ce.SAME_NEAREST_COIN_DIST: -11,
    ce.IS_NOW_SAFE: 10,
    ce.IS_NOW_UNSAFE: -11
}

classic_plus_adil = {
    e.COIN_COLLECTED: 10,
    e.GOT_KILLED: -50,
    e.KILLED_SELF: -50,
    e.INVALID_ACTION: -5,
    e.COIN_FOUND: 3,
    e.KILLED_OPPONENT: 50,
    e.CRATE_DESTROYED: 3,
    ce.DECREASED_NEAREST_COIN_DIST: 0.1,
    ce.INCREASED_NEAREST_COIN_DIST: -0.1,
    #ce.SAME_NEAREST_COIN_DIST: -11,
    ce.IS_NOW_SAFE: 0.5,
    ce.IS_NOW_UNSAFE: -0.5,
    e.WAITED: -0.1,
    e.BOMB_DROPPED: 2
}

# Dynamic Reward Functions

def nearest_coin_distance(game_state: dict):
    if game_state['coins']:
        player_pos = game_state['self'][3]
        min_coin_dist = 99
        for coin_pos in game_state['coins']:
            dist = manhattan_distance(player_pos, coin_pos)
            if dist < min_coin_dist:
                min_coin_dist = dist

        return 1 / min_coin_dist**2 # todo: can change weight factor here

    else:
        return 0




### string to reward set dict

reward_set_strings = {"no_1": game_rewards_no_1,
                      "coin_focus": coin_heaven,
                      "coin_minimal": coin_minimal,
                      "classic": classic,
                      "classic_plus": classic_plus,
                      "classic_plus_adil": classic_plus_adil}

dynamic_rewards_strings = {"coin_dist": nearest_coin_distance}

class RewardGiver:
    __slots__ = ("event_reward_set", "dynamic_reward_set")
    event_reward_set : dict
    dynamic_reward_set : list


    def __init__(self, reward_set: str, dynamic_reward_set = None) -> None:
        self.event_reward_set = reward_set_strings[reward_set]
        if dynamic_reward_set is not None:
            self.dynamic_reward_set = dynamic_reward_set
        else:
            self.dynamic_reward_set = []


    def rewards_from_events(self, events: List[str]) -> int:
        """
        Calculate sum of event-based rewards.
        """

        reward_sum = 0

        # Event based
        for event in events:
            if event in self.event_reward_set:
                reward_sum += self.event_reward_set[event]
        #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        #print(f"Awarded {reward_sum} for events {', '.join(events)}")
        return reward_sum


    def dynamic_rewards(self, old_game_state: dict, action_taken: str, new_game_state: dict):
        """
        Calculate sum of dynamic rewards.
        """
        reward_sum = 0

        if new_game_state is not None: # Last step
            if "coin_dist" in self.dynamic_reward_set:
                reward_sum += nearest_coin_distance(new_game_state)


        return reward_sum