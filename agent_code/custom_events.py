from typing import List
import events as e

class CustomEvents:
    #__slots__ = ("WON_ROUND", "LOST_ROUND")
    PROBABLY_WON = "WON_ROUND"
    PROBABLY_LOST = "LOST_ROUND"
    #BOMB_WHEN_ENEMY_CLOSE
    #if aggressive MOVED_CLOSER_TO_ENEMY

game_rewards = {
    e.COIN_COLLECTED: 100,
    e.KILLED_OPPONENT: 500,
    e.CRATE_DESTROYED: 30,
    e.GOT_KILLED: -300,
    e.INVALID_ACTION: -5,
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_RIGHT: -1,
    e.MOVED_LEFT: -1,
    e.WAITED: -1,
    e.BOMB_DROPPED: 2,
    CustomEvents.PROBABLY_WON: 1000,
    CustomEvents.PROBABLY_LOST: -1000
}

def reward_from_events(events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum