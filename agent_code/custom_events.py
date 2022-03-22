from typing import List

class CustomEvents:
    PROBABLY_WON = "WON_ROUND"
    PROBABLY_LOST = "LOST_ROUND"
    # todo: BOMB_WHEN_ENEMY_CLOSE
    # todo: if aggressive MOVED_CLOSER_TO_ENEMY


def state_to_events(old_game_state: dict, action_taken: str, new_game_state: dict) -> List[str]:
    custom_events = []

    if old_game_state is None:
        # End of round
        won = True
        my_score = new_game_state["self"][1]
        tot_enemy_score = sum([enemy[1] for enemy in new_game_state["others"]])
        points_left = (3 * 5 + 9 * 1) - (my_score + tot_enemy_score)

        # Check if probably won
        for enemy in new_game_state["others"]:
            if my_score < enemy[1] + points_left:
                won = False
                break

        if won:
            custom_events.append(CustomEvents.PROBABLY_WON)
        else:
            custom_events.append(CustomEvents.PROBABLY_LOST)

    else:
        # Additional custom events if not End of Round
        pass

    return custom_events
