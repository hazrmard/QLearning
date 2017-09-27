# QLearning

A Reinforcement Learning library.

*QLearning* implements:

* Model-based,
* Temporal,
* Online/Offline,
* Tabular/Functional

Reinforcement Learning with a focus non-stationary environments.

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
