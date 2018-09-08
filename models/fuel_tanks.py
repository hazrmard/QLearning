import numpy as np



class SixTankModel:
    """
    A model of a six tank system:

    T1  T2  TLA TRA T3  T4

    Each tank is controlled by a valve connected to a shared conduit for fuel
    transfer between tanks. Fuel pumps remove fuel to the engines according to
    pre-defined logic.

    The state space is the fuel level in tanks and the state of valves (on/off).

    [T1  T2  TLA TRA T3  T4 V1  V2  VLA VRA V3  V4]

    The action space is the state of valves.

    [V1  V2  VLA VRA V3  V4]

    Args:

    * fault (int): Tank in which fault/leak is to occur.
    * noise (float): standard deviation of multiplicative gaussian noise applied
    to simulation results.
    * seed (int/None): Random number generator seed for reproducibility.
    """

    def __init__(self, fault=0, noise=0, seed=None):
        self.R = 4.00
        self.F = 8.00
        self.fault = fault
        self.noise = noise
        self.random = np.random.RandomState(seed)


    def set_state(self, S):
        self.tank_1 = S[0]
        self.tank_2 = S[1]
        self.tank_LA = S[2]
        self.tank_RA = S[3]
        self.tank_3 = S[4]
        self.tank_4 = S[5]
        self.DL = S[6]
        self.EL = S[7]
        self.FL = S[8]
        self.FR = S[9]
        self.ER = S[10]
        self.DR = S[11]
    

    def set_action(self, A):
        self.DL = A[0]
        self.EL = A[1]
        self.FL = A[2]
        self.FR = A[3]
        self.ER = A[4]
        self.DR = A[5]


    def run(self, state, action, stepsize=1, **kwargs):
        """
        Simulate tank states.

        Args:

        * state: A tuple/list/array of 12 elements. First 6 are tank levels (0-100),
        last 6 are valve states (0/1)
        * action: A tuple/list/array of 6 elements corresponding to valve states
        for each tank.
        * stepsize: The timestep to simulate over.

        Returns:

        A numpy array of 12 elements - the state vector after simulation.
        """
        self.set_state(state)
        self.set_action(action)
        if ((self.tank_1 + self.tank_2 + self.tank_LA) >= 10 and (self.tank_3 + self.tank_4 + self.tank_RA) >= 10):
            demand_left = 10
            demand_right = 10
        else:
            if ((self.tank_1 + self.tank_2 + self.tank_LA) >= 10):
                demand_right = self.tank_3 + self.tank_4 + self.tank_RA
                if ((self.tank_1 + self.tank_2 + self.tank_LA) >= (10 + 10 - demand_right)):
                    demand_left = 10 + 10 - demand_right
                else:
                    demand_left = self.tank_1 + self.tank_2 + self.tank_LA
            if ((self.tank_3 + self.tank_4 + self.tank_RA) >= 10):
                demand_left = self.tank_1 + self.tank_2 + self.tank_LA
                if ((self.tank_3 + self.tank_4 + self.tank_RA) >= (10 + 10 - demand_left)):
                    demand_right = 10 + 10 - demand_left
                else:
                    demand_right = self.tank_3 + self.tank_4 + self.tank_RA
            if ((self.tank_1 + self.tank_2 + self.tank_LA) < 10 and (self.tank_3 + self.tank_4 + self.tank_RA) < 10):
                demand_left = self.tank_1 + self.tank_2 + self.tank_LA
                demand_right = self.tank_3 + self.tank_4 + self.tank_RA

        if (self.tank_1 >= demand_left):
            pump_1 = demand_left
            pump_2 = 0
            pump_LA = 0
        else:
            pump_1 = self.tank_1
            if (self.tank_2 >= (demand_left - self.tank_1)):
                pump_2 = demand_left - self.tank_1
                pump_LA = 0
            else:
                pump_2 = self.tank_2
                pump_LA = demand_left - self.tank_1 - self.tank_2
        if (self.tank_4 >= demand_right):
            pump_4 = demand_right
            pump_3 = 0
            pump_RA = 0
        else:
            pump_4 = self.tank_4
            if (self.tank_3 >= (demand_right - self.tank_4)):
                pump_3 = demand_right - self.tank_4
                pump_RA = 0
            else:
                pump_3 = self.tank_3
                pump_RA = demand_right - self.tank_3 - self.tank_4
        
        self.tank_1 = self.tank_1 - pump_1
        self.tank_2 = self.tank_2 - pump_2
        self.tank_LA = self.tank_LA - pump_LA
        self.tank_RA = self.tank_RA - pump_RA
        self.tank_3 = self.tank_3 - pump_3
        self.tank_4 = self.tank_4 - pump_4
        
        if ((self.DL + self.EL + self.FL + self.FR + self.ER + self.DR) == 0):
            p = 0
        else:
            p = (self.tank_1 * self.DL + self.tank_2 * self.EL + self.tank_LA * self.FL
                 + self.tank_RA * self.FR + self.tank_3 * self.ER + self.tank_4 * self.DR) / float(self.DL + self.EL + self.FL + self.FR + self.ER + self.DR)
        
        self.tank_1 = self.tank_1 + self.DL * \
            (((p / self.R) - (self.tank_1 / self.R)) * (stepsize)) \
            - ((self.tank_1 / self.F) * (stepsize) if self.fault == 1 else 0)
        self.tank_2 = self.tank_2 + self.EL * \
            (((p / self.R) - (self.tank_2 / self.R)) * (stepsize)) \
            - ((self.tank_2 / self.F) * (stepsize) if self.fault == 2 else 0)
        self.tank_LA = self.tank_LA + self.FL * \
            (((p / self.R) - (self.tank_LA / self.R)) * (stepsize)) \
            - ((self.tank_LA / self.F) * (stepsize) if self.fault == 3 else 0)
        self.tank_RA = self.tank_RA + self.FR * \
            (((p / self.R) - (self.tank_RA / self.R)) * (stepsize)) \
            - ((self.tank_RA / self.F) * (stepsize) if self.fault == 4 else 0)
        self.tank_3 = self.tank_3 + self.ER * \
            (((p / self.R) - (self.tank_3 / self.R)) * (stepsize)) \
            - ((self.tank_3 / self.F) * (stepsize) if self.fault == 5 else 0)
        self.tank_4 = self.tank_4 + self.DR \
            * (((p / self.R) - (self.tank_4 / self.R)) * (stepsize)) \
            - ((self.tank_4 / self.F) * (stepsize) if self.fault == 6 else 0)
        
        noisy = self.random.normal(1, self.noise, 6) * \
                [self.tank_1, self.tank_2, self.tank_LA, self.tank_RA, self.tank_3, self.tank_4]
        return np.concatenate((noisy, action))