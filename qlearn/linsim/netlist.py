"""
This module defines the Netlist class. It is responsible for parsing and
editing the netlist.
"""

try:
    from elements import Element
    from blocks import Block
    from directives import Directive
except ImportError:
    from .elements import Element
    from .blocks import Block
    from .directives import Directive


class Netlist(Block):
    """
    Netlist class parses a circuit netlist and applies modifications. Subclassed
    from Block.

    Args:
        path (str): Path to netlist file. Specify either this OR netlist.
        netlist (tuple/list): Netlist in newline separated definitions.

    Instance Attributes:
        path (str): Filepath of netlist file (default empty string).
        directives (dict): Commands/definitions in the netlist. Stored as
            type: Directive instance.

    Class Attributes:
        prior_directives (tuple): A tuple of directive types that must be put
            before element definitions. Specific to Ahkab library which requires
            'model' directives before element definitions.
        intrinsic_dirctives (tuple): A tuple of directive types that describe
            the circuit itself and not the simulation.
        non_directives (tuple): A tuple of directive types that are not parsed
            as Directive instances. For e.g. .subckt and .ends are parsed as
            Block instances.
    """

    prior_directives = ('model',)
    intrinsic_directives = ('model', 'subckt', 'ends')
    non_directives = ('subckt', 'ends')

    def __init__(self, name, path="", netlist=(), *args, **kwargs):
        self.directives = {}
        self.path = path
        if len(path):
            netlist = self.read_netlist(self.path)
        # elif len(netlist) == 0:
        #     raise AttributeError('Specify either netlist or path.')
        if len(netlist) > 0:
            netlist = self._sanitize('\n'.join(netlist)).split('\n')
            self._parse_directives(netlist)
        super().__init__(name=name, nodes=(), definition=netlist, sanitize=False,\
                            *args, **kwargs)

    @property
    def definition(self):
        """
        Returns the element/block/model definitions without any initial
        condition/plotting/poet-processing directives.
        """
        return self.__str__(dirs=False)


    def __str__(self, dirs=True):
        result = '* Netlist: ' + self.name + '\n'
        # First, only add directive kinds that must come prior to elements
        # Also filter out directive types disallowed if dirs=False
        for kind, directives in self.directives.items():
            if kind in self.__class__.prior_directives and kind != 'end':
                if (not dirs and kind in self.__class__.intrinsic_directives)\
                        or (dirs):
                    for directive in directives:
                        result += str(directive) + '\n'
        # Then add element/block definitions
        result += super().__str__(enclose=False)
        # Finally add the remaining directives
        for kind, directives in self.directives.items():
            if kind not in self.__class__.prior_directives and kind != 'end':
                if (not dirs and kind in self.__class__.intrinsic_directives)\
                        or (dirs):
                    for directive in directives:
                        result += '\n' + str(directive)
        return result + '\n.end'


    def read_netlist(self, path):
        """
        Reads netlist from a file.

        Args:
            path (str): Path to netlist file.
        """
        nfile = open(path, 'r')
        netlist = nfile.readlines()
        nfile.close()
        return netlist


    def _parse_directives(self, netlist):
        """
        Parses the netlist into components and directives.
        """
        for line in netlist:
            if self.is_directive(line):
                self.add_directive(line)


    def add_directive(self, directive):
        """
        Appends a directive to the netlist. The last directive is always
        '.end'.

        Args:
            directive (str/Directive): A directive line or instance.
        """
        if not isinstance(directive, Directive):
            directive = Directive(definition=directive)
        if directive.kind in self.directives:
            self.directives[directive.kind].append(directive)
        else:
            self.directives[directive.kind] = [directive]


    def is_directive(self, elem):
        """
        Checks whether a line is a directive.

        Args:
            elem (str/Element): A line in the netlist or an Element instance.

        Returns:
            boolean -> True if line is directive.
        """
        return str(elem)[0] == '.' \
                and str(elem).split()[0][1:] not in self.__class__.non_directives
