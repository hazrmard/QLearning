"""
This module creates a testbench for the learning algorithm which illustrates
its progress graphically. The problem is a tolopogy where the objective is to
get to the lowest altitude possible.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from qlearner import QLearner



class TestBench:
    """
    The testbench class simulates the learning procedure implemented by the
    Qlearner class graphically for illustration. It generates a geographical
    system - a square grid where each point has a height. The goal states are
    coordinates with lower heights. Possible actions/transitions are movements
    to adjacent points. There are two state variables: x and y coords. Each
    instance automatically generates transition and reward matrices that can
    be used by a QLearner instance.

    Args:
        size (int): The size of each side of the topology (size * size points).
        seed (int): The seed for the random number generator.
        method (str): Method for generating topology. Default='fault'.
        goals (int): Number of goal states. Default = size.
        qlearner (QLearner): QLearner instance which has called learn(). Defaults
            to None in which case a Qlearner instance with default arguments is
            generated with any extra keyword arguments passed on.
        **kwargs: A sequence of keyword arguments to instantiate the qlearner
            if one is not provided. Any keywords except tmatrix, rmatrix, and
            goals which are provided by the TestBench.

    Instance Attributes:
        topology (2D ndarray): A square array of heights defining the system.
            Heights are stored as [y, x] coordinates in line with the array
            indexing convention.
        size (int): Size of topology (length of side).
        states (int): Number of possible states/positions in the system.
        actions (2D ndarray): An array where each row defines an action by a
            change in topology coordinates. For e.g. [1,0] is move-up.
        path (list): Sequence of coordinates (y, x) tuples taken on topology to
            reach goal state.
        tmatrix (2D ndarray): The transition matrix.
        rmatrix (2D ndarray): The reward matrix.
        goals (list): List of goal state numbers (coords encoded into int).
        num_goals (int): Number of goal states.
        qlearner (QLearner): QLearner instance. The learn() function must be
            called before visualizing the learned policy function.
        fig (plt.figure): A matplotlib figure instance storing all plots.
        topo_ax (Axes3D): An axis object storing the 3D plots.
        topo_surface (Poly3DCollection): Contains plotted surface.
        path_line (Line3D): Stores the path taken on topology after learning.
    """

    plot_num = 0

    def __init__(self, size=10, seed=0, method='fault', goals=-1, qlearner=None,
                 **kwargs):
        np.random.seed(seed)
        # qlearning params
        self.topology = np.zeros((size, size))
        self.size = size
        self.states = size * size
        self.actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.path = []
        self.tmatrix = np.array([])
        self.rmatrix = np.array([])
        self.goals = []
        self.num_goals = size if goals < 0 else goals
        self.qlearner = qlearner

        # plotting variables
        self.fig_num = self.__class__.plot_num
        self.fig = plt.figure(self.fig_num)
        self.__class__.plot_num += 1
        self.topo_ax = self.fig.add_subplot(111, projection='3d')
        self.topo_surface = None
        self.path_line = None

        self.create_topology(method)
        self.generate_trg()
        if self.qlearner is None:
            self.qlearner = QLearner(self.rmatrix, self.goals, self.tmatrix, \
                            **kwargs)


    def episode(self, start=None, interactive=False, show=True, limit=-1):
        """
        Run a single episode from the provided qlearner. The episode starts at
        coordinates 'start' and ends when it reaches a goal state. Calls the
        self.qlearner.recommend() function to get sequence of actions to take
        based on the learned Q-Matrix.

        Args:
            start (list/tuple/ndarray): y and x coordinates to start from. If
                None, generates random coordinates.
            interactive (bool): If true, prompts after each step, otherwise
                draws the entire path at once.
            show (bool): If true, shows a figure with the topology etc. Else,
                proceeds silently while storing state history (as coords) in
                self.path.
            limit (int): Maximum number of steps in episode before quitting.
                Defaults to self.size*self.size. Only applies with interactve=
                False.
        """
        if start is None:
            start = (np.random.randint(self.size), np.random.randint(self.size))
        self.path.append(tuple(start))
        limit = self.size**2 if limit <= 0 else limit
        current = self.coord2state(start)
        if interactive:
            pass
        else:
            iteration = 0
            while current not in self.goals and iteration < limit:
                iteration += 1
                action = self.qlearner.recommend(current)
                current = self.tmatrix[current, action]
                self.path.append(self.state2coord(current))
            if show:
                self.show_topology(True)


    def create_topology(self, method='fault'):
        """
        Creates a square height map based on one of several terrain generation
        algorithms. The topology is stored in self.topology (2D ndarray), where
        self.topology[y, x] = height at coordinate (x, y).

        Args:
            method (str): The algorithm to use. Default='fault'.
        """
        if method == 'fault':
            self._fault_algorithm(int(np.random.rand() * 200))


    def show_topology(self, block=False):
        """
        Draws a surface plot of the topology, marks goal states, and any episode
        up to its current progress.

        Args:
            block (bool): If true, halts execution of following statements as
                long as the plot is open.
        """
        # Plot 3d topology surface
        x, y = np.meshgrid(np.linspace(0, self.size-1, self.size),\
                            np.linspace(0, self.size-1, self.size))
        z = self.topology.reshape(x.shape)
        self.topo_surface = self.topo_ax.plot_surface(x, y, z, cmap='gist_earth')
        # Plot goal states
        gc = [self.state2coord(i) for i in self.goals]  # goal coords
        gz = [self.topology[g[0], g[1]] for g in gc]
        gx = [g[1] for g in gc]
        gy = [g[0] for g in gc]
        self.topo_ax.scatter(gx, gy, gz)
        # Plot path
        px = [p[1] for p in self.path]
        py = [p[0] for p in self.path]
        pz = [self.topology[p[0], p[1]] for p in self.path]
        self.path_line = self.topo_ax.plot(px, py, pz)
        # Display figure
        plt.show(block=block)
        if block:
            self.fig.clear()
            plt.close(self.fig_num)



    def _fault_algorithm(self, iterations):
        """
        The Fault Algorithm generates a terrain by drawing a line of a random
        gradient through the grid, and be elevating points on one side, and
        lowering points on the other side for some number of iterations.
        """
        angle = np.random.rand(iterations) * 2 * np.pi
        a = np.sin(angle)
        b = np.cos(angle)
        disp = (np.random.rand() / iterations) * (np.arange(iterations)[::-1] + 1)
        for i in range(iterations):
            cx, cy = np.random.rand(2) * self.size
            for x in range(self.size):
                for y in range(self.size):
                    self.topology[y, x] += disp[i] \
                                            if -a[i]*(x-cx) + b[i]*(y-cy) > 0 \
                                            else -disp[i]


    def generate_trg(self):
        """
        Calculates goal states and creates transition & reward matrices. Where
        tmatrix[state, action] points to index of next state. And
        rmatrix[state, action] is the reward for taking that action. State is
        he encoded state number from coords2state. The transitions roll over:
        i.e going right from the right-most coordinate takes to the left-most
        coordinate.
        """
        tmatrix = np.zeros((self.states, len(self.actions)), dtype=int)
        rmatrix = np.zeros((self.states, len(self.actions)))
        # updating goal states
        goal_state = np.argsort(np.ravel(self.topology))[:self.num_goals]

        for i in range(self.states):
            coords = self.state2coord(i)
            # updating r and t matrices
            for j, action in enumerate(self.actions):
                next_coord = coords + action
                next_coord[next_coord < 0] += self.size
                next_coord[next_coord >= self.size] -= self.size
                tmatrix[i, j] = self.coord2state(next_coord)

                rmatrix[i, j] = self.topology[coords[0], coords[1]] \
                                - self.topology[next_coord[0], next_coord[1]]
        self.tmatrix = tmatrix
        self.rmatrix = rmatrix
        self.goals = goal_state


    def coord2state(self, coord):
        """
        Encodes coordinates into a state number that can be used as an index.
        Essentially converts Base_size coordinates into Base_10.

        Args:
            coord (tuple/list/ndarray): 2 coordinates [row, column]

        Returns:
            An integer representing coordinates.
        """
        return int(self.size * coord[0] + coord[1])


    def state2coord(self, state):
        """
        Converts a state number into two coordinates. Essentially converts
        Base_10 state into Base_size coordinates.

        Args:
            state (int): An integer representing a state.

        Returns:
            A 2 element tuple of [row, column] coordinates.
        """
        return (int(state/self.size) % self.size, state % self.size)
