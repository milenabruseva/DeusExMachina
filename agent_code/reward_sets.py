from typing import List
import events as e
from .custom_events import CustomEvents as ce


### Reward Sets as dicts

game_rewards_no_1 = {
    e.COIN_COLLECTED: 10,
    e.COIN_FOUND: 3,
    e.KILLED_OPPONENT: 50,
    e.CRATE_DESTROYED: 3,
    e.GOT_KILLED: -50,
    e.KILLED_SELF: -50,
    e.INVALID_ACTION: -5,
    e.MOVED_UP: -0.1,
    e.MOVED_DOWN: -0.1,
    e.MOVED_RIGHT: -0.1,
    e.MOVED_LEFT: -0.1,
    e.WAITED: -0.1,
    e.BOMB_DROPPED: 2,
    ce.PROBABLY_WON: 100,
    ce.PROBABLY_LOST: -100
}



### string to reward set dict

reward_set_strings = {"no_1": game_rewards_no_1}



class RewardGiver:
    __slots__ = "reward_set"
    reward_set : dict


    def __init__(self, reward_set: str) -> None:
        self.reward_set = reward_set_strings[reward_set]


    def rewards_from_events(self, events: List[str]) -> int:
        """
        Calculate total reward.
        """
        reward_sum = 0
        for event in events:
            if event in self.reward_set:
                reward_sum += self.reward_set[event]
        # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        return reward_sum

