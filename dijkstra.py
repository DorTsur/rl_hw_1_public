from puzzle import *
from planning_utils import *
import heapq
import datetime


def dijkstra(puzzle):
    '''
    apply dijkstra to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # the fringe is the queue to pop items from
    fringe = [(0, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}
    ignore = set()

    if initial == goal:
        return prev

    while len(fringe) > 0:
        # stopping condition - when we reach the goal state
        _, current_state = heapq.heappop(fringe)
        possible_actions = current_state.get_actions()

        for action in possible_actions:
            new_state = current_state.apply_action(action)
            if new_state == goal:
                distances[new_state.to_string()] = 1 + distances[current_state.to_string()]
                prev[new_state.to_string()] = current_state
                return prev
            ###
            if new_state.to_string() in concluded or new_state.to_string() in ignore:
                continue
            ###
            # if new_state.to_string() in concluded:
            #     continue
            if new_state.to_string() not in distances.keys():
                distances[new_state.to_string()] = float('inf')
            # d = new_state.get_manhattan_distance(current_state) + distances[current_state.to_string()]
            d = 1 + distances[current_state.to_string()]
            if d < distances[new_state.to_string()]:
                distances[new_state.to_string()] = d
                prev[new_state.to_string()] = current_state
                heapq.heappush(fringe, (d, new_state))
                # print("d = {}".format(d))
                ###
                ignore.add(new_state.to_string())
                ###

        # fringe = ignore_duplicates(fringe)
        concluded.add(current_state.to_string())

        # assert False
    return prev

def ignore_duplicates(q):
    ignore = set()
    q_new = []
    # this is how we ignore items:
    while len(q) > 0:
        current_priority, current_item = heapq.heappop(q)
        if current_item.to_string() not in ignore:
            ignore.add(current_item.to_string())
            heapq.heappush(q_new, (current_priority, current_item))
    return q_new


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = dijkstra(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l',
        'd','u','d','u',
        'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    # actions = ['r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r']
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
