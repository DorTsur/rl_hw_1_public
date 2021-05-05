import numpy as np
import matplotlib.pyplot as plt
from cartpole_cont import CartPoleContEnv

############################
#     Bashar Huleihel
#        Dor Tsur
############################

def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    A = np.zeros((4,4))
    A[0,1] = 1
    A[1,2] = pole_mass/cart_mass*g
    A[2,3] = 1
    A[3,2] = g/pole_length*(1+pole_mass/cart_mass)

    A = np.eye(4) + dt*A
    return A


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    B = np.zeros((4, 1))

    B[1] = 1 / cart_mass
    B[3] = 1 / (cart_mass * pole_length)

    B = B * dt

    return B


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action, np.matrix of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # TODO - you first need to compute A and B for LQR
    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    # TODO - Q and R should not be zero, find values that work, hint: all the values can be <= 1.0
    # Our Q:
    # for sections 2-4
    w_1 = 0.6
    w_2 = 1
    w_3 = 0.01

    # for section 5
    # w_1 = 0.6
    # w_2 = 1
    # w_3 = 0.95

    Q = np.matrix([
        [w_1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, w_2, 0],
        [0, 0, 0, 0]
    ])

    # Our R:
    R = np.matrix([w_3])

    # initialization:
    Ps = [Q]
    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    Ks = []

    # calculation of the values over a loop:
    for i in range(cart_pole_env.planning_steps):
        Ks.append( -np.linalg.inv(np.transpose(B)@Ps[i]@B+R)@np.transpose(B)@Ps[i]@A )
        us.append( Ks[i]@xs[i] )
        Ps.append( Q + np.transpose(A) @ Ps[i] @ A + np.transpose(A) @ Ps[i - 1] @ B @ Ks[i] )
        xs.append( A@xs[i]+B@us[i] )

    # reverse the lists as we calculate in the "opposite" time direction:
    Ks.reverse()
    us.reverse()
    Ps.reverse()
    xs.reverse()

    # check valid dimensions:
    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


if __name__ == '__main__':
    section = 1
    if section == 3:
        # calculation with a iterations over several values of theta_init:
        theta_unstable = np.pi*0.2996  # for sections 2-4
        # theta_unstable = np.pi*0.115  # for section 5
        init_theta = [0.1*np.pi, theta_unstable/2, theta_unstable]  # for section 3
        # init_theta = [0.1, 0.00000001, 0.000000001]  # for section 4
        theta_trajectory = []

        for i, theta_ in enumerate(init_theta):
            theta_trajectory.append([])
            env = CartPoleContEnv(initial_theta=theta_)

            tau = env.tau
            num_steps = env.planning_steps

            # start a new episode
            actual_state = env.reset()
            env.render()
            # use LQR to plan controls
            xs, us, Ks = find_lqr_control_input(env)
            # run the episode until termination, and print the difference between planned and actual
            is_done = False
            iteration = 0
            is_stable_all = []
            while not is_done:
                # print the differences between planning and execution time
                predicted_theta = xs[iteration].item(2)
                actual_theta = actual_state[2]

                # saving between [-pi,pi]
                statement = True
                while statement:
                    if actual_theta > np.pi:
                        actual_theta = actual_theta - 2*np.pi
                    elif actual_theta < -np.pi:
                        actual_theta = actual_theta + 2*np.pi
                    else:
                        statement = False

                # actual_state[2] = actual_theta
                theta_trajectory[i].append(actual_theta)
                predicted_action = us[iteration].item(0)
                actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
                print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
                # apply action according to actual state visited
                # make action in range
                actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
                actual_action = np.array([actual_action])
                actual_state, reward, is_done, _ = env.step(actual_action)  # using with actual actions
                # actual_state, reward, is_done, _ = env.step(predicted_action*np.ones_like(actual_action))   # using with predicted action
                is_stable = reward == 1.0
                is_stable_all.append(is_stable)
                env.render()
                iteration += 1
            env.close()
            # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
            valid_episode = np.all(is_stable_all[-100:])
            # print if LQR succeeded
            print('valid episode: {}'.format(valid_episode))

        # Plotting results for several theta values:
        t = np.arange(0.0, tau*(num_steps), tau)
        fig, ax = plt.subplots()
        ax.set(xlabel='time (s)', ylabel='pole angle',
               title='pole angle vs. time')
        ax.grid()
        for j in range(len(init_theta)):
            y = theta_trajectory[j]
            ax.plot(t, y, label="theta = {}".format(init_theta[j]))

        ax.legend()
        fig.savefig("cartpole_angle_evolution.png")
        plt.show()

    else:
        # calculation without loops:
        env = CartPoleContEnv(initial_theta=np.pi * 0.1)

        # print the matrices used in LQR
        print('A: {}'.format(get_A(env)))
        print('B: {}'.format(get_B(env)))

        # start a new episode
        actual_state = env.reset()
        env.render()

        # use LQR to plan controls
        xs, us, Ks = find_lqr_control_input(env)
        # run the episode until termination, and print the difference between planned and actual
        is_done = False
        iteration = 0
        is_stable_all = []

        while not is_done:  # run over a single episode:
            # print the differences between planning and execution time
            predicted_theta = xs[iteration].item(2)
            actual_theta = actual_state[2]
            predicted_action = us[iteration].item(0)
            actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
            print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
            actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
            actual_action = np.array([actual_action])
            actual_state, reward, is_done, _ = env.step(actual_action)
            is_stable = reward == 1.0
            is_stable_all.append(is_stable)
            env.render()
            iteration += 1
        env.close()

        valid_episode = np.all(is_stable_all[-100:])
        # print if LQR succeeded
        print('valid episode: {}'.format(valid_episode))

