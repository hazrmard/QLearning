"""
This module defines the Netlist class. It is responsible for parsing and
editing the netlist.
"""


class Netlist:
    """
    Netlist class parses a circuit netlist and applies modifications.

    Args:
        path (str): Path to netlist file. Specify either this OR netlist.
        netlist (str): The entire netlist string.

    Attributes:
        netlist (list): A list of netlist comments/elements/directives.
        path (str): Filepath of netlist file (default empty string).
        elements (list): Element declarations in the netlist.
        directives (list): Commands/definitions in the netlist.
    """

    def __init__(self, path="", netlist=()):
        self.elements = []
        self.directives = []
        self.netlist = [n.lower() for n in netlist]
        self.path = path
        if len(path):
            self.read_netlist(self.path)
        elif len(self.netlist):
            self.parse_netlist()
        else:
            raise AttributeError('Specify either netlist or path.')


    def read_netlist(self, path, parse=True):
        """
        Reads netlist from a file.

        Args:
            path (str): Path to netlist file.
            parse (bool): Whether to parse netlist as well.
        """
        nfile = open(path, 'r')
        self.netlist = nfile.readlines()
        self.netlist = [x.lower().strip() for x in self.netlist]
        nfile.close()
        if parse:
            self.parse_netlist()


    def parse_netlist(self):
        """
        Parses the netlist into components and directives.
        """
        for line in self.netlist:
            if self.is_directive(line):
                self.directives.append(line)
            elif self.is_element(line):
                self.elements.append(line)


    def modify_element(self, elem, line):
        """
        Modifies the netlist based on the directive(s) passed. Looks up
        matching components in the netlist and overwrites/appends directive.

        Args:
            elem (str): Element id in netlist.
            line (str): Element definition to be written to netlist.
        """
        pass


    def delete(self, elem):
        """
        Finds and deletes element from netlist.

        Args:
            elem (str): Element id in netlist.
        """
        self.elements = [x for x in self.elements if self.get_id(x) != elem.lower()]


    def compile_netlist(self):
        """
        Compiles all modifications into a new netlist. Does not preserve
        comments etc.

        Returns:
            A netlist string.
        """
        self.netlist = self.elements + self.directives
        return '\n'.join(self.netlist)


    def is_element(self, line):
        """
        Checks whether a line defines a component.

        Returns:
            boolean -> True if line is component.
        """
        return line[0] != '*' and line[0] != '.'


    def is_directive(self, line):
        """
        Checks whether a line is a directive.

        Returns:
            boolean -> True if line is directive.
        """
        return line[0] == '.'


    def get_id(self, line):
        """
        Returns the element id from a netlist definition.

        Args:
            line (str): A single element declaration from the netlist.

        Returns:
            A string containing element id.
        """
        if self.is_element(line):
            return (line.split(' ')[0]).strip()
        else:
            raise ValueError('Line is not an element definition.')


    def get_nodes(self, line):
        """
        Returns the positive and negative nodes element is connected to.

        Args:
            line (str): A line of element definition from the netlist.

        Returns:
            A tuple of two node ids (str): (positive node, negative node)
        """
        split = line.split(' ')
        return (split[1].strip(), split[2].strip())


    def get_definition(self, elem):
        """
        Given element id, returns the netlist line defining it.

        Args:
            elem (Str): Element id in netlist.

        Returns:
            A single line of element's definition in netlist.
            If not found, returns empty string.
        """
        for line in self.elements:
            if elem.lower() == self.get_id(line):
                return line
        return ""
