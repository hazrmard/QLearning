try:
    from netlist import Netlist
    from simulate import Simulator
    from system import System
    from flags import FlagGenerator
    from elements import *
    from directives import Directive
except ImportError:
    from .netlist import Netlist
    from .simulate import Simulator
    from .system import System
    from .flags import FlagGenerator
    from .elements import *
    from .directives import Directive
