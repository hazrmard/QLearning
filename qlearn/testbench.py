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
    to adjacent points. There are two state variables: x and y coords.
    It visualizes the path a QLearner would take after learning from the reward
    matrix generated for that topology.

    Args:
        size (int): The size of each side of the topology.
        seed (int): The seed for the random number generator.
        method (str): Method for generating topology. Default='fault'.
        goals (int): Number of goal states. Default = size.

    Instance Attributes:
        topology (2D ndarray): A square array of heights defining the system.
            Heights are stored as [y, x] coordinates in line with the array
            indexing convention.
        size (int): Size of topology (length of side).
        states (int): Number of possible states/positions in the system.
        actions (2D ndarray): An array where each row defines an action by a
            change in topology coordinates. For e.g. [1,0] is move-up.
        path (list): Sequence of coordinates [y, x] taken on topology to reach
            goal state.
        tmatrix (2D ndarray): The transition matrix.
        rmatrix (2D ndarray): The reward matrix.
        goals (list): List of goal state numbers (coords encoded into int).
        num_goals (int): Number of goal states.
        fig (plt.figure): A matplotlib figure instance storing all plots.
        topo_ax (Axes3D): An axis object storing the 3D plots.
        topo_surface (Poly3DCollection): Contains plotted surface.
        path_line (Line3D): Stores the path taken on topology after learning.
    """

    plot_num = 0

    def __init__(self, size=10, seed=0, method='fault', goals=-1):
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

        # plotting variables
        self.fig_num = self.__class__.plot_num
        self.fig = plt.figure(self.fig_num)
        self.__class__.plot_num += 1
        self.topo_ax = self.fig.add_subplot(111, projection='3d')
        self.topo_surface = None
        self.path_line = None

        self.create_topology(method)
        self.generate_trg()


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
        Draws a surface plot of the topology.

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
        # Display figure
        plt.show(block=block)
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
        tmatrix = np.zeros((self.states, len(self.actions)))
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
        return self.size * coord[0] + coord[1]


    def state2coord(self, state):
        """
        Converts a state number into two coordinates. Essentially converts
        Base_10 state into Base_size coordinates.

        Args:
            state (int): An integer representing a state.

        Returns:
            A 2 element ndarray of [row, column] coordinates.
        """
        return np.array((int(state/self.size) % self.size, state % self.size))
