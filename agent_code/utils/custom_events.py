from typing import List

from agent_code.utils.features import get_blast_coords, is_path_possible_astar

class CustomEvents:
    DECREASED_NEAREST_COIN_DIST = "DECREASED_NEAREST_COIN_DIST"
    SAME_NEAREST_COIN_DIST = "SAME_NEAREST_COIN_DIST"
    INCREASED_NEAREST_COIN_DIST = "INCREASED_NEAREST_COIN_DIST"
    ENEMY_TOOK_NEAREST_COIN = "ENEMY_TOOK_NEAREST_COIN"
    PROBABLY_WON = "WON_ROUND"
    PROBABLY_LOST = "LOST_ROUND"
    IS_NOW_SAFE = "IS_NOW_SAFE"
    IS_NOW_UNSAFE = "IS_NOW_UNSAFE"
    NO_CHANGE_IN_TILE_SAFETY = "NO_CHANGE_IN_TILE_SAFETY"

def state_to_events(old_game_state: dict, action_taken: str, new_game_state: dict, killed_opponents_scores,
                    end_of_round) -> List[str]:
    custom_events = []

    if end_of_round:
        custom_events.append(won_or_lost_event(killed_opponents_scores, new_game_state))

    else: # Additional custom events if not End of Round
        # Coin distance events
        if new_game_state['coins'] and old_game_state['coins']:
            custom_events.append(coin_event(old_game_state, new_game_state))

        # State changed from safe/unsafe events
        custom_events.append(safe_or_unsafe_event(old_game_state, new_game_state))

    return list(filter(None, custom_events))


def won_or_lost_event(killed_opponents_scores, new_game_state):
    self_score = new_game_state["self"][1]
    remaining_points_possible = new_game_state["remaining_coins"] + len(new_game_state["others"]) * 5
    max_opponent_scores_possible = {}
    for opp in new_game_state["others"]:
        max_opponent_scores_possible[opp[0]] = opp[1] + remaining_points_possible - 5
    killed_opponents_winning = len(
        {opponent: score for (opponent, score) in killed_opponents_scores.items() if score >= self_score})
    living_opponents_can_win = len(
        {opponent: score for (opponent, score) in max_opponent_scores_possible.items() if score >= self_score})
    if not killed_opponents_winning and not living_opponents_can_win:
        return CustomEvents.PROBABLY_WON
    else:
        return CustomEvents.PROBABLY_LOST

def coin_event(old_game_state, new_game_state):
    new_min_coin_dist, new_nearest_coin_pos, new_player_pos, old_min_coin_dist, old_nearest_coin_pos = calculate_coin_distances(
        new_game_state, old_game_state)
    # Add appropriate event
    if not is_path_possible_astar(new_game_state["field"], new_player_pos, new_nearest_coin_pos):
        return None

    if new_nearest_coin_pos == old_nearest_coin_pos:
        if new_min_coin_dist < old_min_coin_dist:
            return CustomEvents.DECREASED_NEAREST_COIN_DIST
        elif new_min_coin_dist == old_min_coin_dist:
            return CustomEvents.SAME_NEAREST_COIN_DIST
        else:
            return CustomEvents.INCREASED_NEAREST_COIN_DIST
    else:
        if new_player_pos == old_nearest_coin_pos:  # player took nearest coin
            pass
        else:
            if new_min_coin_dist == old_min_coin_dist:  # player potentially switches between two coin,
                # or enemy took last nearest coin while a coin with same old distance was available
                return CustomEvents.SAME_NEAREST_COIN_DIST
            elif new_min_coin_dist > old_min_coin_dist:  # enemy took last nearest coin while
                # the next nearest coin was further away
                return CustomEvents.ENEMY_TOOK_NEAREST_COIN
            else:  # this shouldn't be possible to happen
                return None


def safe_or_unsafe_event(old_game_state, new_game_state):
    was_safe = is_agent_safe(old_game_state)
    now_safe = is_agent_safe(new_game_state)
    state_changed = was_safe != now_safe
    if state_changed and now_safe:
        return CustomEvents.IS_NOW_SAFE
    elif state_changed and not now_safe:
        return CustomEvents.IS_NOW_UNSAFE
    else:
        return CustomEvents.NO_CHANGE_IN_TILE_SAFETY

def is_agent_safe(game_state):
    player_pos = game_state["self"][3]
    bombs = []
    blast_coords = []

    if len(game_state["bombs"]):
        for i in range(len(game_state["bombs"])):
            bombs.append(game_state["bombs"][i][0])

    for bomb in bombs:
        blast_coords.extend(get_blast_coords(bomb, game_state["field"]))

    if player_pos in blast_coords:
        return False
    return True

def calculate_coin_distances(new_game_state, old_game_state):
    new_player_pos = new_game_state['self'][3]
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
    return new_min_coin_dist, new_nearest_coin_pos, new_player_pos, old_min_coin_dist, old_nearest_coin_pos

def manhattan_distance(x1: tuple[int, int], x2: tuple[int, int]):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])