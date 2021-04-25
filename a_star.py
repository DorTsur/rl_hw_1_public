from puzzle import *
from planning_utils import *
import heapq
import datetime
import matplotlib.pyplot as plt


def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
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
    Number_states = 0
    alpha = 1
    while len(fringe) > 0:
        # stopping condition - when we reach the goal state
        _, current_state = heapq.heappop(fringe)
        possible_actions = current_state.get_actions()
        for action in possible_actions:
            Number_states = Number_states + 1
            new_state = current_state.apply_action(action)
            if new_state == goal:
                distances[new_state.to_string()] = 1 + distances[current_state.to_string()]
                prev[new_state.to_string()] = current_state
                print('Number of visited states:{}'.format(Number_states))
                return prev
            ###
            if new_state.to_string() in concluded or new_state.to_string() in ignore:
                continue
            ###
            # if new_state.to_string() in concluded:
            #     continue
            if new_state.to_string() not in distances.keys():
                distances[new_state.to_string()] = float('inf')
            d = 1 + distances[current_state.to_string()]
            if d < distances[new_state.to_string()]:
                h_ns = alpha * new_state.get_manhattan_distance(goal)
                distances[new_state.to_string()] = d
                prev[new_state.to_string()] = current_state
                heapq.heappush(fringe, (d + h_ns, new_state))
                ignore.add(new_state.to_string())
        concluded.add(current_state.to_string())
        # assert False
    print('Number of visited states:{}'.format(Number_states))
    return prev

def a_star_alpha(puzzle, alpha):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
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
    Number_states = 0
    while len(fringe) > 0:
        # stopping condition - when we reach the goal state
        _, current_state = heapq.heappop(fringe)
        possible_actions = current_state.get_actions()
        for action in possible_actions:
            Number_states = Number_states + 1
            new_state = current_state.apply_action(action)
            if new_state == goal:
                distances[new_state.to_string()] = 1 + distances[current_state.to_string()]
                prev[new_state.to_string()] = current_state
                print('Number of visited states:{}'.format(Number_states))
                return prev, Number_states
            ###
            if new_state.to_string() in concluded or new_state.to_string() in ignore:
                continue
            ###
            # if new_state.to_string() in concluded:
            #     continue
            if new_state.to_string() not in distances.keys():
                distances[new_state.to_string()] = float('inf')
            d = 1 + distances[current_state.to_string()]
            if d < distances[new_state.to_string()]:
                h_ns = alpha * new_state.get_manhattan_distance(goal)
                distances[new_state.to_string()] = d
                prev[new_state.to_string()] = current_state
                heapq.heappush(fringe, (d + h_ns, new_state))
                ignore.add(new_state.to_string())
        concluded.add(current_state.to_string())
        # assert False
    print('Number of visited states:{}'.format(Number_states))
    return prev, Number_states

def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan

def solve_alpha(puzzle, alpha):
    # compute mapping to previous using dijkstra
    prev_mapping, states = a_star_alpha(puzzle, alpha)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return states


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'u', 'd', 'u', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)

    # our test:
    # initial_state = State('8 6 7\n2 5 4\n3 0 1')
    # goal_state = State('1 2 3\n4 5 6\n7 8 0')

    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))

    # our test of alpha:
    # states = []
    # alpha_vec = [0,0.01,0.25,0.5,0.75,1,2.5,5,7.5,10,25,50,75,100,250,500,750,1000,2500,5000,10000,10000000]
    # for alpha in alpha_vec:
    #     states.append(solve_alpha(puzzle,alpha))
    #
    # plt.figure()
    # plt.plot(alpha_vec, states)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("Number of visited states vs. values of alpha", fontsize=16)
    # plt.xlabel("alpha")
    # plt.ylabel("Number of visited states")
    # plt.show()

