"""
This module defines the Netlist class. It is responsible for parsing and
editing the netlist.
"""

try:
    from elements import Element
    from blocks import Block
except ImportError:
    from .elements import Element
    from .blocks import Block


class Netlist(Block):
    """
    Netlist class parses a circuit netlist and applies modifications. Subclassed
    from Block.

    Args:
        path (str): Path to netlist file. Specify either this OR netlist.
        netlist (tuple/list): Netlist in newline separated elements.

    Attributes:
        path (str): Filepath of netlist file (default empty string).
        directives (list): Commands/definitions in the netlist.
    """

    def __init__(self, name, path="", netlist=(), *args, **kwargs):
        self.directives = []
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
        result += super().__str__(enclose=False)
        if len(self.directives):
            result += '\n' + '\n'.join(self.directives)
        return result


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
                self.directives.append(line)


    def add_directive(self, directive):
        """
        Appends a directive to the netlist. The last directive is always
        '.end'. dir is inserted at second-last position in self.directives.

        Args:
            directive (str): A directive line.
        """
        self.directives.insert(-1, directive)
