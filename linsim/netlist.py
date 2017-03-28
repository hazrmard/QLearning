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
        netlist (tuple/list): Netlist in newline separated elements.

    Instance Attributes:
        path (str): Filepath of netlist file (default empty string).
        directives (dict): Commands/definitions in the netlist. Stored as
            type: Directive instance.

    Class Attributes:
        prior_directives (tuple): A tuple of directive types that must be put
            before element definitions. Specific to Ahkab library which requires
            'model' directives before element definitions.
    """

    prior_directives = ('model',)

    def __init__(self, name, path="", netlist=(), *args, **kwargs):
        self.directives = {}
        self.path = path
        if len(path):
            netlist = self.read_netlist(self.path)
        elif len(netlist) == 0:
            raise AttributeError('Specify either netlist or path.')
        netlist = self._sanitize(''.join(netlist)).split('\n')
        self._parse_directives(netlist)
        super().__init__(name=name, nodes=(), definition=netlist, sanitize=False,\
                            *args, **kwargs)


    def __str__(self):
        result = '* Netlist: ' + self.name + '\n'
        for kind, directives in self.directives.items():
            if kind in self.__class__.prior_directives and kind != 'end':
                for directive in directives:
                    result += str(directive) + '\n'
        result += super().__str__(enclose=False)
        for kind, directives in self.directives.items():
            if kind not in self.__class__.prior_directives and kind != 'end':
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
