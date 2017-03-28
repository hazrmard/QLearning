"""
This module defines the Directive class and its derivatives. A Directive
informs the type of simulation, defines element models, and provodes control
flow in the netlist.
"""

try:
    from elements import Element
except ImportError:
    from .elements import Element


class Directive(Element):
    """
    A directive defines an instruction in the netlist informing the simulation.
    It starts with a period '.'. Of the format:
        .TYPE [ARG1 ARG2...] [PARAM1=VALUE1,...]
    Subclassed from Element.

    Args:
        definition (str): The netlist line containing the directive (see format
            above).
        OR:
        *args, **kwargs: Positional any keyword arguments in the same order as
            found in the directive string.

    Instance Attributes:
        kind: Returns the name of the directive minus the leading period found
            in self.name.
    """

    prefix = '.'
    num_nodes = 0

    @property
    def kind(self):
        """
        Returns directive name, ignoring prefix.
        """
        return self.name[1:]


    def _verify(self, args):
        if self.__class__.prefix.lower() != args[0][:len(self.__class__.prefix)]:
            raise ValueError('Incorrect directive format.')
