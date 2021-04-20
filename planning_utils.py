def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    state = goal_state
    while prev[state.to_string()] is not None:
        prev_state = prev[state.to_string()]
        actions = prev_state.get_actions()
        for a in actions:
            if prev_state.apply_action(a) == state:
                result.append((prev_state, a))
                state = prev_state
                break
    # remove the following line and complete the algorithm
    # assert False
    result.reverse()
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
