from typing import List

class CustomEvents:
    DECREASED_NEAREST_COIN_DIST = "DECREASED_NEAREST_COIN_DIST"
    SAME_NEAREST_COIN_DIST = "SAME_NEAREST_COIN_DIST"
    INCREASED_NEAREST_COIN_DIST = "INCREASED_NEAREST_COIN_DIST"
    ENEMY_TOOK_NEAREST_COIN = "ENEMY_TOOK_NEAREST_COIN"
    PROBABLY_WON = "WON_ROUND"
    PROBABLY_LOST = "LOST_ROUND"
    # todo: BOMB_WHEN_ENEMY_CLOSE
    # todo: if aggressive MOVED_CLOSER_TO_ENEMY, implement dynamically as reward that is proportional to 1/dist**2


def state_to_events(old_game_state: dict, action_taken: str, new_game_state: dict) -> List[str]:
    custom_events = []

    if new_game_state is None:
        # End of round
        won = True
        my_score = old_game_state["self"][1]
        tot_enemy_score = sum([enemy[1] for enemy in old_game_state["others"]])
        points_left = (3 * 5 + 9 * 1) - (my_score + tot_enemy_score)

        # Check if probably won
        for enemy in old_game_state["others"]:
            if my_score < enemy[1] + points_left:
                won = False
                break

        if won:
            custom_events.append(CustomEvents.PROBABLY_WON)
        else:
            custom_events.append(CustomEvents.PROBABLY_LOST)

    else:
        # Additional custom events if not End of Round

        # Coin distance calculation
        if new_game_state['coins'] and old_game_state['coins']:
            new_player_pos= new_game_state['self'][3]
            old_player_pos = old_game_state['self'][3]
            new_min_coin_dist = 99
            new_nearest_coin_pos = None
            old_min_coin_dist = 99
            old_nearest_coin_pos = None
            for coin_pos in new_game_state['coins']:
                dist = manhattan_distance(new_player_pos, coin_pos)
                if dist < new_min_coin_dist:
                    new_min_coin_dist = dist
                    new_nearest_coin_pos = coin_pos
            for coin_pos in old_game_state['coins']:
                dist = manhattan_distance(old_player_pos, coin_pos)
                if dist < old_min_coin_dist:
                    old_min_coin_dist = dist
                    old_nearest_coin_pos = coin_pos

            # Add appropriate event
            if new_nearest_coin_pos == old_nearest_coin_pos:
                if new_min_coin_dist < old_min_coin_dist:
                    custom_events.append(CustomEvents.DECREASED_NEAREST_COIN_DIST)
                elif new_min_coin_dist == old_min_coin_dist:
                    custom_events.append(CustomEvents.SAME_NEAREST_COIN_DIST)
                else:
                    custom_events.append(CustomEvents.INCREASED_NEAREST_COIN_DIST)
            else:
                if new_player_pos == old_nearest_coin_pos: # player took nearest coin
                    pass
                else:
                    if new_min_coin_dist == old_min_coin_dist: # player potentially switches between two coin,
                        # or enemy took last nearest coin while a coin with same old distance was available
                        custom_events.append(CustomEvents.SAME_NEAREST_COIN_DIST) # todo: is this the appropriate event?
                    elif new_min_coin_dist > old_min_coin_dist: # enemy took last nearest coin while
                        # the next nearest coin was further away
                        custom_events.append(CustomEvents.ENEMY_TOOK_NEAREST_COIN)
                    else: # this shouldn't be possible to happen
                        pass

    return custom_events

def manhattan_distance(x1: tuple[int, int], x2: tuple[int, int]):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])