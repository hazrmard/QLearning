"""
Tests for the linsim package.
"""

import os
from elements import Element
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


@test
def test_element_class():
    """Test element class for parsing definitions"""

    # Set up
    def1 = "R100 N1 0 100k"
    def2 = "C25 N1 N2 25u"
    def3 = "G1 N3 n2 n1 0 table=(0 0, 10 100)"

    # Test 1: checking definition parsing for default Element class
    elem = Element(def1)
    assert [str(n) for n in elem.nodes] == ['n1', '0'], 'Nodes incorrectly parsed.'
    assert elem.value == ['100k',], 'Value incorrectly parsed.'

    elem = Element(def2)
    assert [str(n) for n in elem.nodes] == ['n1', 'n2'], 'Nodes incorrectly parsed.'
    assert elem.value == ['25u',], 'Value incorrectly parsed.'

    elem = Element(def3)
    assert [str(n) for n in elem.nodes] == ['n3', 'n2'], 'Nodes incorrectly parsed.'
    assert elem.value == ['n1', '0'], 'Value incorrectly parsed.'
    assert hasattr(elem, 'table'), 'Param=Value pair incorrectly parsed.'


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
    test_element_class()
    test_netlist_io()
    print('\n==========\n')
    print('Tests passed:\t' + str(TESTS_PASSED))
    print('Total tests:\t' + str(NUM_TESTS))
    print()
