**Note**: The `dev` branch of this repository has been migrated to a separate repository [`Agents`](https://github.com/hazrmard/Agents). This `legacy` branch contains reference code for IFAC Safeproccess 2018 paper: *Comparison of Model Predictive and Reinforcement Learning Methods for Fault Tolerant Control by Ibrahim Ahmed, Gautam Biswas, and Hamed Khorasgani*.

---

# QLearning

A Reinforcement Learning library.

*QLearning* implements:

* Model-based,
* Temporal,
* Online/Offline,
* Tabular/Functional

Reinforcement Learning with a focus non-stationary environments.

| <img src="/img/topology.png" alt="RL on a topology with faults" width="420"/> | <img src="/img/tanks.gif" alt="Visualization of a fuel tank system modelled as a circuit" width="420"/> |
|:---:|:---:|
|RL on a topology with faults. | Visualization of a fuel tank system modelled as a circuit. |

Features include:

* A modelling framework based on Netlist syntax and a circuit simulator. An
environment can be programmatically/statically represented as an electrical circuit. Or the framework can be used to define custom elements and manipulate a modular system to represent changes in the environment.

* Modular structure. A custom environment model can be easily used. All learner classes can be subclassed and still maintain compatibility.

* Learning in descrete and continuous state spaces. Spaces represented by a finite combination of flags (e.g multidimensional grids) can be encoded into integers or decoded into vectors for tabular representation and functional approximations.

* Flexible value function learning. Learner classes provide errors which can be used to train a custom approximation to the value function.

* A built-in testbench to provide continuous/descrete environments to test and visualize learning.

The default learning algorithm is n-step tree back-up with variable step sizes.

## Installation

`Qlearning` requires:

* Python 3
* Dependencies:
  * flask
  * numpy
  * ahkab
  * matplotlib

1. Install [Python 3](https://www.python.org/downloads/).
2. [Download](https://github.com/hazrmard/QLearning/archive/legacy.zip)/clone this repository.
3. Open terminal/console and navigate to this cloned repository. Then use the package manager `pip` to install dependencies:

```
cd Qlearning
pip install -r requirements.txt
```


## Demo

The `models/` directory comes with a 6-tank model for a C-130 plane's fuel tanks.

### SixTankModel

The code ships with a simple model of 6 fuel tanks in a plane. The state space is 12 dimensional (6 tank levels + 6 valve states). The action space is 6 dimensional (6 valve states).

```
    T1  T2  TLA | TRA T3  T4
```

Each tank is controlled by a valve connected to a shared conduit for fuel transfer between tanks. Fuel pumps remove fuel to the engines according to pre-defined logic.

The state space is the fuel level in tanks (0-100) and the state of valves (on/off).

```
    [T1  T2  TLA TRA T3  T4 V1  V2  VLA VRA V3  V4]
```

The action space is the state of valves.

```
[V1  V2  VLA VRA V3  V4]
```

The `SixTanksModel` class is a programmatic model of the tanks. The class has a `run()` method which simulates the next state, given current tank levels, valve states, and action. For more details, see `models/fuel_tanks.py`.

```python
from models import SixTankModel

tanks = SixTankModel()

next_state = tanks.run(state=(10, 20, 10, 30, 14, 50, 0, 0, 1, 1, 0, 1),
                       action=(1, 1, 1, 0, 0, 0))
```


### Visualizing demo

There are several demos illustrating various control approaches to the problem of fuel transfer. The process can be visualized  in a browser. The three scripts `tankssimple.py`, `tanksnetlist.py` and `tankscustom.py` use different approaches to modelling the system:

* `tankssimple.py`: Models tanks as a circuit of capacitors and current sources. Creates a netlist at runtime to simulate.
* `tankssimple.py`: Models tanks as a circuit of capacitors and resistors. Loads a netlist file `models/tanks.netlist`.
* `tankscustom.py`: Models tanks as a set of formulae written in python. Uses the `SixTankModel` class defined in `models/fuel_tanks.py`.

```
# to see help message about arguments
python tankscustom.py -h

# visualize tank model behavior
# python tankscustom.py -x
```

The visualization can be viewed at http://localhost:5000
