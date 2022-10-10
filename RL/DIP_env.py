import json
import math
from pathlib import Path

import gym
from casadi import *

from RL.rewards import DoubleCartpoleReward

EQ_PATH = Path(__file__).parent / "equations.json"


class double_inverted_pendulum:
    def __init__(self, file_name_equations=EQ_PATH, dt=0.02, maneuver="stabilization"):
        """
        Create double inverted pendulum

        Args:
            file_name_equations: system dynamic
            dt:
            maneuver: type of maneuver. Swing_up or stabilization. Default maneuver = "stabilization"
        """
        # """
        #     dt (float): Time step [s].
        #     mc (float): Cart mass [kg].
        #     mp1 (float): First link mass [kg].
        #     mp2 (float): Second link mass [kg].
        #     l1 (float): First link length [m].
        #     l2 (float): Second link length [m].
        #     mu (float): Coefficient of friction [dimensionless].
        #     g (float): Gravity acceleration [m/s^2].
        # """
        # TODO integrate parameters in env
        self.dt = dt
        self.maneuver = maneuver
        self.theta_threshold_radians = 100000 * 2 * math.pi / 360
        self.x_threshold = 10
        self.max_input = 4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.max_input,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
        ])

        self.action_space = gym.spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32)

        # self.action_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=np.full(6, -float('inf'), dtype=np.float32),
        #                                         high=np.full(6, float('inf'), dtype=np.float32), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-high,
                                                high=high, dtype=np.float32)

        # sundials solver
        self.solver = self.read_model(file_name_equations)

        # init system state (hanging down)

        # self.init_state = np.zeros((6, 1))

        if self.maneuver == "stabilization":
            # for stabilization :
            self.init_state = np.array([0, 0.1, -0.1, 0, 0, 0])
            self._max_episode_steps = 200  # max steps for one episode
        if self.maneuver == "swing_up":
            # for Swing up
            self.init_state = np.array([0, 0, 0, 0, 0, 0])
            self.init_state[2] = np.pi
            self.init_state[4] = np.pi
            self._max_episode_steps = 200  # max steps for one episode
        # curr state of the system
        self.state = self.init_state

        # state vector if pendulum is fully erected and stable
        self.goal = np.zeros((6, 1))

        # steps simulated
        self.steps_simulated = 0
        self.viewer = None
        self.display = None

    def step(self, action):
        # simulate system
        self.state = np.asarray(self.solver(x0=self.state, p=action)["xf"])

        # increase step counter
        self.steps_simulated += 1
        # original format x = (xc, xc_dot, phi1, phi1_dot, phi2, phi2_dot).T
        obs = (self.state.T).squeeze()  # for new reward class

        # second approach (from https://github.com/fregu856/CS-229/blob/master/cartpole.py)
        # r = - (10 * normalize_angle(obs[2]) + 10 * normalize_angle(obs[1]))
        # r = - (10 * math.cos(obs[2]) + 10 * math.cos(obs[1]))

        reward_func = DoubleCartpoleReward()
        r = reward_func(obs, [action])

        # check if pendulum is stable and erected e.g. check of error between current position and goal is small enough
        if self.maneuver == "stabilization":
            done = obs[0] < -self.x_threshold \
                   or obs[0] > self.x_threshold \
                   or self.steps_simulated > self._max_episode_steps \
                   or obs[2] > 90 * 2 * np.pi / 360 \
                   or obs[2] < -90 * 2 * np.pi / 360

        if self.maneuver == "swing_up":
            done = obs[0] < -self.x_threshold \
                   or obs[0] > self.x_threshold \
                   or self.steps_simulated > self._max_episode_steps

        done = bool(done)

        return obs, r.squeeze().item(), done, {}

    def reset(self):
        self.state = self.init_state
        self.steps_simulated = 0
        return self.state.squeeze()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return None

        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        # carty = 100 # TOP OF CART
        # #
        carty = 300  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.2
        cartwidth = 50.0
        cartheight = 30.0
        # return render_v1(self.viewer, self.state)
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.2, .6, .4)
            self.poletrans2 = rendering.Transform(translation=(0, polelen - 5))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.poletrans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            self.axle2 = rendering.make_circle(polewidth / 2)
            self.axle2.add_attr(self.poletrans2)
            self.axle2.add_attr(self.poletrans)
            self.axle2.add_attr(self.carttrans)
            self.axle2.set_color(.1, .5, .8)
            self.viewer.add_geom(self.axle2)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
        state = self.state
        cartx = state.item(0) * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(state.item(2))
        self.poletrans2.set_rotation((state.item(4) - state.item(2)))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def read_model(self, file_name):
        """
        Wrapper function for importing sympy equation in casadi.
        Second order ODE are reduced to a system of first order ODES. Afterwards a sundials solver is initialized and
        returned.
        """

        # get equation in string format
        with open(file_name) as json_file:
            sympy_sol = json.load(json_file)

        # define symbolic casadi variables used in ODEs
        x_c = SX.sym('x_c')
        phi_1 = SX.sym('phi_1')
        phi_2 = SX.sym('phi_2')
        xdot_c = SX.sym('xdot_c')
        phidot_1 = SX.sym('phidot_1')
        phidot_2 = SX.sym('phidot_2')
        xddot_c = SX.sym('xddot_c')

        x = SX.sym("x", 6)
        u = SX.sym('u', 1)

        # replace all "sin" "cos" functions in equation string with numpy functions
        for var in sympy_sol:
            sympy_sol[var] = sympy_sol[var].replace("sin", "np.sin")
            sympy_sol[var] = sympy_sol[var].replace("cos", "np.cos")

        # create casadi function for each ODE this will make the reduction of order easier
        xddot_c_cas_symb = eval(sympy_sol["xddot_c"])
        xddot_c_fun = Function("xddot_c", [phi_1, phi_2, xdot_c, phidot_1, phidot_2, u], [xddot_c_cas_symb])

        phiddot_1_cas_symb = eval(sympy_sol["phiddot_1"])
        phiddot_1_fun = Function("phiddot_1", [phi_1, phi_2, xdot_c, phidot_1, phidot_2, u], [phiddot_1_cas_symb])

        phiddot_2_cas_symb = eval(sympy_sol["phiddot_2"])
        phiddot_2_fun = Function("phiddot_2", [phi_1, phi_2, xdot_c, phidot_1, phidot_2, u], [phiddot_2_cas_symb])

        # create system of first order ODEs
        xdot = np.array([[x[1]],  # xdot_1
                         [xddot_c_fun(x[2], x[4], x[1], x[3], x[5], u)],  # xdot_2
                         [x[3]],  # xdot_3
                         [phiddot_1_fun(x[2], x[4], x[1], x[3], x[5], u)],  # xdot_3
                         [x[5]],  # xdot_4
                         [phiddot_2_fun(x[2], x[4], x[1], x[3], x[5], u)]])  # xdot_4

        # dict for options and ode needed for solver object
        ode = {'x': x, 'ode': xdot, 'p': u}
        opts = {'tf': self.dt}

        # return solver object
        return integrator('F', 'idas', ode, opts)

    def state_dict(self):
        pass

    def seed(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()


def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2 * np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2 * np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    import os

    output_folder = Path(__file__).parent.parent / "runs"
    results_folder = os.path.join(output_folder, "plots")
    num_samples = 200
    dt = 0.01
    env = double_inverted_pendulum(dt=dt)
    env.reset()
    save_dict = {}
    obs_list = []
    action_list = []
    dones = []
    u = 1.
    t = 0

    writer = SummaryWriter(log_dir=os.path.join(results_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    done = False
    while not done:
        u = env.action_space.sample()
        obs, r, done, _ = env.step(u.item())
        writer.add_scalar("states/postion_x", obs[0], t)
        writer.add_scalar("states/velocity_x", obs[1], t)
        writer.add_scalar("states/phi_1", obs[2], t)
        writer.add_scalar("states/phi_2", obs[4], t)
        writer.add_scalar("reward", r, t)
        env.render()
        t += 1
    # save_dict["tt"] = [obs_list, action_list]
    # torch.save(save_dict, os.path.join(results_folder, str(k)+'eval_dynamics.pth.tar'))
