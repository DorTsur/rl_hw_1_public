from puzzle import *
from planning_utils import *
import heapq
import datetime

############################
#     Bashar Huleihel
#        Dor Tsur
############################

def dijkstra(puzzle):
    '''
    apply dijkstra to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    initial = puzzle.start_state
    goal = puzzle.goal_state
    fringe = [(0, initial)]
    concluded = set()
    distances = {initial.to_string(): 0}
    prev = {initial.to_string(): None}
    ignore = set()

    if initial == goal:  # verify that we do not begin at the terminal state
        return prev
    Number_states = 0

    while len(fringe) > 0:
        _, current_state = heapq.heappop(fringe)
        possible_actions = current_state.get_actions()

        for action in possible_actions:
            Number_states = Number_states + 1
            new_state = current_state.apply_action(action)
            if new_state == goal:  # algorithm stopping condition - when we reach the goal state
                distances[new_state.to_string()] = 1 + distances[current_state.to_string()]
                prev[new_state.to_string()] = current_state
                print('Number of visited states:{}'.format(Number_states))
                return prev
            if new_state.to_string() in concluded or new_state.to_string() in ignore:  # the condition in 8-puzzle for skipping
                continue
            if new_state.to_string() not in distances.keys():
                distances[new_state.to_string()] = float('inf')  # define new encountered step distance to be infinity
            d = 1 + distances[current_state.to_string()]  # distance calculation from initial state
            if d < distances[new_state.to_string()]:  # distance updating condition
                distances[new_state.to_string()] = d
                prev[new_state.to_string()] = current_state
                heapq.heappush(fringe, (d, new_state))
                ignore.add(new_state.to_string())

        concluded.add(current_state.to_string())

    return prev


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
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)

    # our "hard" test:
    # initial_state = State('8 6 7\n2 5 4\n3 0 1')
    # goal_state = State('1 2 3\n4 5 6\n7 8 0')

    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
