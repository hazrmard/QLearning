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
environment can be programmatically/statically represented as an electrical
circuit. Or the framework can be used to define custom elements and manipulate
a modular system to represent changes in the environment.

* Modular structure. A custom environment model can be easily used. All learner
classes can be subclassed and still maintain compatibility.

* Learning in descrete and continuous state spaces. Spaces represented by a
finite combination of flags (e.g multidimensional grids) can be encoded into
integers or decoded into vectors for tabular representation and functional
approximations.

* Flexible value function learning. Learner classes provide errors which can
be used to train a custom approximation to the value function.

* A built-in testbench to provide continuous/descrete environments to test
and visualize learning.

The default learning algorithm is n-step tree back-up with variable step sizes
(see `references.md`) for more details.

*Note*: This project is under active development. This `README` is just a brief
overview of functionality. See top-level `.py` files for demonstrations of use.
The code is also extensively annotated for documentation.
