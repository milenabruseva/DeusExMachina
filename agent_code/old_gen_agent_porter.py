def old_state_dict_from_new(agent, state) -> dict:
    state["arena"]  = state.pop("field")

    old_self = state["self"]
    state["self"] = (old_self[3][0], old_self[3][1], old_self[0], old_self[1], old_self[2])

    state["train"] = agent.train

    for idx, other in enumerate(state["others"]):
        old_other = other
        state["others"][idx] = (old_other[3][0], old_other[3][1], old_other[0], old_other[1], old_other[2])

    for idx, bomb in enumerate(state["bombs"]):
        state["bombs"][idx] = (bomb[0][0], bomb[0][1], bomb[1])

    state["explosions"] = state.pop("explosion_map")

    return state
