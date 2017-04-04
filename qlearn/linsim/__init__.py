try:
    from netlist import Netlist
    from simulate import Simulator
    from system import System
    from flags import FlagGenerator
except ImportError:
    from .netlist import Netlist
    from .simulate import Simulator
    from .system import System
    from .flags import FlagGenerator
