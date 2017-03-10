"""
Tests for the linsim package.
"""

import os
from flags import FlagGenerator
from elements import Element
from elements import ElementMux
from nodes import Node
from blocks import Block
from netlist import Netlist
from simulate import Simulator
from system import System

NUM_TESTS = 0
TESTS_PASSED = 0
QLEARNER = None

def test(func):
    """
    Decorator for test cases.

    Args:
        func (function): A test case function object.
    """
    global NUM_TESTS
    NUM_TESTS += 1
    def test_wrapper(*args, **kwargs):
        """
        Wrapper that calls test function.

        Args:
            desc (str): Description of test.
        """
        print(func.__doc__.strip(), end='\t')
        try:
            func(*args, **kwargs)
            global TESTS_PASSED
            TESTS_PASSED += 1
            print('PASSED')
        except Exception as ex:
            print('FAILED: ' + str(ex))

    return test_wrapper


#@test
def test_flag_generator():
    """Test flag generation from states"""

    # Set up
    flags = [4, 3, 2]
    states = 4 * 3 * 2

    # Test 1: Instantiation
    gen = FlagGenerator(*flags)
    assert gen.states == states, "Flag state calculation failed."

    # Test 2: Basis conversion
    assert gen.convert_basis(10, 2, 5) == [1, 0, 1], "Decimal to n-ary failed."
    assert gen.convert_basis(6, 10, (2, 4)) == [1, 6], "N-ary to decimal failed."
    assert gen.convert_basis(2, 8, (1, 0, 1)) == [5], "N-ary to n-ary failed."
    assert gen.convert_basis(10, 2, [1, 0]) == [1, 0, 1, 0], "Decimal to n-ary failed."

    # Test 3: Encoding and decoding
    assert gen.decode(12) == [2, 0, 0], 'Decoding failed.'
    assert gen.encode(*gen.decode(12)) == 12, 'Encoding decoding mismatch.'


@test
def test_node_class():
    """Test node class for hashing"""

    # Set up
    n1 = Node(1)
    n2 = Node(2)
    n1x = Node(1)
    ndict = {}

    # Test 1: Testing hashing into dictionary
    ndict[n1] = 1
    ndict[n2] = 2
    assert ndict.get(n1) == 1, 'Improper node hashing.'
    assert ndict.get(n2) == 2, 'Improper node hashing.'

    # Test 2: Testing retrieval and equality checks
    assert ndict.get(n1x) == ndict.get(n1), 'Node equality failed. Bad hashing.'
    assert n1 == n1.name, 'Node equality failed with strings.'


@test
def test_element_class():
    """Test element class for parsing definitions"""

    # Set up
    def1 = "R100 N1 0 100k"
    def2 = "C25 N1 N2 25u"
    def3 = "G1 N3 n2 n1 0 table=(0 0, 10 100)"

    # Test 1: checking definition parsing for default Element class
    elem = Element(definition=def1)
    assert [str(n) for n in elem.nodes] == ['n1', '0'], 'Nodes incorrectly parsed.'
    assert elem.value == '100k', 'Value incorrectly parsed.'

    elem = Element(definition=def2)
    assert [str(n) for n in elem.nodes] == ['n1', 'n2'], 'Nodes incorrectly parsed.'
    assert elem.value == '25u', 'Value incorrectly parsed.'

    elem = Element(definition=def3)
    assert [str(n) for n in elem.nodes] == ['n3', 'n2'], 'Nodes incorrectly parsed.'
    assert elem.value == 'n1 0', 'Value incorrectly parsed.'
    assert hasattr(elem, 'table'), 'Param=Value pair incorrectly parsed.'
    assert elem.param('table') == '(0 0,10 100)', 'Param fetching failed.'

    # Test 2: checking argument/keyword parsing for default Element class
    elem = Element('T1', 10, 12, 100, k1=1, K2=2)
    assert str(elem) == 't1 10 12 100 k1=1 k2=2', 'Arg parsing failed.'


@test
def test_element_mux():
    """Test the element multiplexer"""

    # Set up
    class a:
        prefix = 'a'
        def __init__(self, *args, **kwargs):
            pass
    class b(a):
        prefix = 'b'
    class bc(b):
        prefix = 'bc'
    class x(a):
        prefix = 'x'
    def_b = 'b200 blah blah'
    def_bc = 'bcb1 blah blah blah'
    def_a = 'a7 asdjaa alskdj'
    def_other = 'j20 asd knwe'

    # Test 1: Testing mux generation
    mux = ElementMux(root=a)
    assert set(mux.prefix_list) == set(['b', 'bc', 'x']), \
                'Element mux generation failed.'

    # Test 2: Testing multiplexing
    assert mux.mux(def_b).prefix == 'b', 'Incorrect multiplexing.'
    assert mux.mux(def_bc).prefix == 'bc', 'Incorrect multiplexing.'
    assert mux.mux(def_a).prefix == 'a', 'Incorrect multiplexing.'
    assert mux.mux(def_other).prefix == 'a', 'Incorrect multiplexing.'

    # Test 3: Testing mux editing
    class k(a):
        prefix = 'k'
    mux.add('k', k)
    assert mux.mux('k100 blah blah').prefix == 'k', 'Mux addition failed.'
    mux.remove('k')
    assert ('k' not in mux.prefix_list and 'k' not in mux._mux), \
        'Mux deletion failed.'


@test
def test_block_class():
    """Test block parsing"""

    # Set up
    block1_def = """blah blah
some more blah"""
    block1 = """.subckt block1 1 2 n3 n12
""" + block1_def + """
.ends block1"""

    block2_def = """blah blah
some more blah"""
    block2 = """.subckt block2 1 2 n3 n12
""" + block2_def + """
.ends block2"""

    block_str = block1 + """
asd
asd
""" + block2 + """

asd
asd
asd"""
    block_defs = block_str.split('\n')

    # Test 1: Instantiation
    block = Block('test', ('1', 'n2', 'node3'), block_defs)

    # Test 2: Parsing correctness
    assert len(block.blocks) == 2, 'Incorrect number of blocks detected.'
    assert block.blocks['block1'].name == 'block1', "Nested block name parsing failed."
    assert block.blocks['block2'].name == 'block2', "Nested block name parsing failed."
    assert block.blocks['block1'].definition == block1_def, "Nested block def parsing failed."
    assert block.blocks['block2'].definition == block2_def, "Nested block def parsing failed."


@test
def test_netlist_io():
    """Test Netlist class for reading/parsing"""

    # Set up
    net = ("C1 0 T1 1mF\n"
           "R1 T1 N001 1k\n"
           "G1 N001 0 T1 0 1 table=(0 0, 0.1 1m)\n"
           ".ic V(T1)=10V\n"
           ".tran 0 15s 0 1m uic\n"
           ".backanno\n"
           ".end")
    tfile = open('test.net', 'w')
    tfile.write(net)
    tfile.close()

    # Test 1: reading netlist file
    ninstance = Netlist(path="test.net")
    assert ninstance.compile_netlist() == net.lower(), "Netlist read incorrectly."

    # Test 2: Parsing netlist

    # Finalizing
    os.remove('test.net')





if __name__ == '__main__':
    print()
    test_flag_generator()
    test_node_class()
    test_element_class()
    test_element_mux()
    test_block_class()
    test_netlist_io()
    print('\n==========\n')
    print('Tests passed:\t' + str(TESTS_PASSED))
    print('Total tests:\t' + str(NUM_TESTS))
    print()
