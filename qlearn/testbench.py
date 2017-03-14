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
    system - a square grid where each point has a height. The goal state is
    a coords with lower heights. There are two state variables: x and y coords.
    It visualizes the path a QLearner would take after learning from the reward
    matrix generated for that topology.

    Args:
        size (int): The size of each side of the topology.
        seed (int): The seed for the random number generator.
        method (str): Method for generating topology. Default='fault'.
        goals (int): Number of goal states. Default = size.

    Instance Attributes:
        topology (2D ndarray): A square array of heights defining the system.
        size (int): Size of topology (length of side).
        states (int): Number of possible states/positions in the system.
        actions (2D ndarray): An array where each row defines an action by a
            change in topology coordinates. For e.g. [1,0] is move-right.
        fig (plt.figure): A matplotlib figure instance storing all plots.
        topo_ax (subplot): An axis object storing the 3D surface plot.
        tmatrix (2D ndarray): The transition matrix.
        rmatrix (2D ndarray): The reward matrix.
        goals (list): List of goal state numbers (coords encoded into int).
        num_goals (int): Number of goal states.
    """

    plot_num = 0

    def __init__(self, size=10, seed=0, method='fault', goals=-1):
        self.topology = np.zeros((size, size))
        self.size = size
        self.states = size * size
        self.actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.tmatrix = np.array([])
        self.rmatrix = np.array([])
        self.goals = []
        self.num_goals = size if goals < 0 else goals
        self.fig = plt.figure(self.__class__.plot_num)
        self.__class__.plot_num += 1
        self.topo_ax = self.fig.add_subplot(111, projection='3d')
        np.random.seed(seed)

        self.create_topology(method)
        self.generate_trg()


    def create_topology(self, method='fault'):
        """
        Creates a square height map based on one of several terrain generation
        algorithms. The topology is stored in self.topology (2D ndarray).

        Args:
            method (str): The algorithm to use. Default='fault'.
        """
        if method == 'fault':
            self._fault_algorithm(int(np.random.rand() * 1000))


    def show_topology(self, block=False):
        """
        Draws a surface plot of the topology.

        Args:
            block (bool): If true, halts execution of following statements as
                long as the plot is open.
        """
        x, y = np.meshgrid(np.linspace(0, self.size-1, self.size),\
                            np.linspace(0, self.size-1, self.size))
        z = self.topology.reshape(x.shape)
        self.topo_ax.plot_surface(x, y, z)
        plt.show(block=block)



    def _fault_algorithm(self, iterations):
        angle = np.random.rand(iterations) * np.pi
        a = np.sin(angle)
        b = np.cos(angle)
        max_fault_dist = np.sqrt(2) * self.size
        c = (np.random.rand(iterations) * (2*self.size)) - self.size
        disp = (np.random.rand() / iterations) * (np.arange(iterations)[::-1] + 1)
        for i in range(iterations):
            for x in range(self.size):
                for y in range(self.size):
                    self.topology[y, x] += disp[i] if -a[i]*x + b[i]*y > b[i]*c[i] else -disp[i]


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
