"""
Tests for the linsim package.
"""

import os
import numpy as np
try:
    from flags import FlagGenerator
    from elements import *
    from directives import Directive
    from nodes import Node
    from blocks import Block
    from netlist import Netlist
    from simulate import Simulator
except ImportError:
    from .flags import FlagGenerator
    from .elements import *
    from .directives import Directive
    from .nodes import Node
    from .blocks import Block
    from .netlist import Netlist
    from .simulate import Simulator

NUM_TESTS = 0
TESTS_PASSED = 0

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
def test_flag_generator():
    """Test flag generation from states"""

    # Set up
    flags = [4, 3, 2]
    flags2 = [[-1, 1], 2]
    flags3 = [[-5, 20, 4]]
    states = 4 * 3 * 2
    states2 = 3 * 2
    states3 = 20

    # Test 1: Instantiation
    gen = FlagGenerator(*flags)
    gen2 = FlagGenerator(*flags2)
    gen3 = FlagGenerator(*flags3)
    assert gen.num_states == states, "Flag state calculation failed."
    assert gen2.num_states == states2, "Flag state calculation failed."
    assert gen3.num_states == states3, "Flag state calculation failed."

    # Test 2: Basis conversion
    assert np.array_equal(gen.convert_basis(10, 2, 5), [0, 1, 0, 1]), "Decimal to n-ary failed."
    assert np.array_equal(gen.convert_basis(6, 10, (2, 4)), [1, 6]), "N-ary to decimal failed."
    assert np.array_equal(gen.convert_basis(2, 8, (1, 0, 1)), [0, 5]), "N-ary to n-ary failed."
    assert np.array_equal(gen.convert_basis(10, 2, [1, 0]), [0, 1, 0, 1, 0]), "10-ary to n-ary failed."

    # Test 3: Encoding and decoding
    assert np.array_equal(gen.decode(12), [2, 0, 0]), 'Decoding failed.'
    assert gen.encode(*gen.decode(12)) == 12, 'Encoding decoding mismatch.'
    assert np.array_equal(gen2.decode(0), [-1, 0]), 'Decoding failed.'
    assert gen2.encode(*gen2.decode(0)) == 0, 'Encoding decoding mismatch.'
    assert np.array_equal(gen3.decode(1), [-4.5]), 'Decoding failed.'
    assert gen3.encode(*gen3.decode(1)) == 1, 'Encoding decoding mismatch.'


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
    def1 = "R100 N1 0 10e5"
    def2 = "C25 N1 N2 25e-3"
    def3 = "G1 N3 n2 n1 0 table=(0 1e-1, 10 100)"

    # Test 1: checking definition parsing for default Element class
    elem = Element(definition=def1)
    assert [str(n) for n in elem.nodes] == ['n1', '0'], 'Nodes incorrectly parsed.'
    assert elem.value == '10e5', 'Value incorrectly parsed.'

    elem = Element(definition=def2)
    assert [str(n) for n in elem.nodes] == ['n1', 'n2'], 'Nodes incorrectly parsed.'
    assert elem.value == '25e-3', 'Value incorrectly parsed.'

    elem = Element(definition=def3)
    assert [str(n) for n in elem.nodes] == ['n3', 'n2'], 'Nodes incorrectly parsed.'
    assert elem.value == 'n1 0', 'Value incorrectly parsed.'
    # assert hasattr(elem, 'table'), 'Param=Value pair incorrectly parsed.'
    assert elem.param('table') == '(0 1e-1,10 100)', 'Param fetching failed.'
    elem.param('table', '')
    assert elem.param('table') is None, 'Param deletion failed.'

    elem = Element(num_nodes=3, definition=def3)
    assert elem.nodes == ['n3', 'n2', 'n1'], 'Custom node numbers failed.'

    # Test 2: checking argument/keyword parsing for default Element class
    elem = Element('T1', 10, 12, 100, k1=1, K2=2)
    assert str(elem) == 't1 10 12 100 k1=1 k2=2' or \
                        str(elem) == 't1 10 12 100 k2=2 k1=1',\
                        'Arg parsing failed.'

    # Test 3: checking preset Element subclasses
    r = 'r1 n1 0 1e4'
    c = 'c1 n1 0 1e4'
    l = 'l1 n1 0 1e4'
    v1 = 'v1 n1 0 type=vdc vdc=10'
    v2 = 'v2 n1 0 type=sin VO=10 VA=1.2 FREQ=500e3 TD=1e-9 THETA=0'
    v3 = 'v3 n1 0'
    func = lambda t: 2*t
    e = 'e1 n1 n2 sn1 sn2 10'
    f = 'f1 n1 n2 elem1 10'
    g = 'g1 n1 n2 sn1 sn2 10'
    h = 'h1 n1 n2 elem1 10'
    m = 'm1 n1 n2 n3 n4 model w=1 l=2'
    d = 'd1 n1 n2 model AREA=1 T=1 OFF=False'

    R = Resistor(definition=r)
    C = Capacitor(definition=c)
    L = Inductor(definition=l)
    V1 = VoltageSource(definition=v1)
    V2 = VoltageSource(definition=v2)
    V3 = VoltageSource(definition=v3, function=func)
    E = VoltageControlledVoltageSource(definition=e)
    F = CurrentControlledCurrentSource(definition=f)
    G = VoltageControlledCurrentSource(definition=g)
    H = CurrentControlledVoltageSource(definition=h)
    M = Transistor(definition=m)
    D = Diode(definition=d)

    assert R.value == 1e4, 'Resistor value not parsed.'
    assert C.value == 1e4, 'Capacitor value not parsed.'
    assert L.value == 1e4, 'Inductor value not parsed.'
    assert V1.param('type') == 'vdc', 'Voltage source type not parsed.'
    assert V1.param('vdc') == 10., 'Voltage source value not parsed.'
    assert V2.param('va') == 1.2, 'Voltage source wave param not parsed.'
    assert V3.param('function') is None and callable(V3.function), \
            'Custom function not parsed'
    assert E.value == 10., 'Dependent source alpha not parsed.'
    assert F.value == ('elem1', 10.), 'Dependent source tuple not parsed.'
    assert G.value == 10., 'Dependent source alpha not parsed.'
    assert H.value == ('elem1', 10.), 'Dependent source tuple not parsed.'
    assert M.param('w') == 1., 'Transistor param not parsed.'
    assert D.param('off') == 'false', 'Diode boolean not parsed.'

@test
def test_directive_class():
    """Test netlist directive parsing"""

    # Set up
    dir1 = '.tran 0s 10s'
    dir2 = '.ic V(n1)=10 i(10) =500mA'
    dir3 = '.end'

    # Test 1: Instantiation
    ins1 = Directive(definition=dir1)
    ins2 = Directive(definition=dir2)
    ins3 = Directive(definition=dir3)

    # Test 2: Type checking
    assert ins1.kind == 'tran' and ins2.kind == 'ic' and ins3.kind == 'end',\
                        'Incorrect directive types.'

    # Test 3: String conversion
    assert str(ins1) == dir1.lower(), 'Directive string conversion 1 failed.'
    assert str(ins2) == '.ic v(n1)=10 i(10)=500ma' or \
            str(ins2) == '.ic i(10)=500ma v(n1)=10',\
            'Directive string conversion 2 failed.'


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
    mux = ElementMux(root=a, leave=('x',))
    assert set(mux.prefix_list) == set(['b', 'bc']), 'Element mux generation failed.'

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
    block1_def = ("y1 1 2 45\n"
                  "y2 2 3 50")
    block1 = ".subckt block1 1 2 n3 n12\n" \
             + block1_def + "\n" \
             + ".ends block1"

    block2_def = ("y3 4 5 100\n"
                  "y4 3 4 whatever")
    block2 = ".subckt block2 1 2 n3 n12\n" \
                + block2_def + "\n" \
                ".ends block2"

    elems1 = "y6 2 4 90\n" \
            + "yLEM45 1 2 64"
    elems2 = "ys1 s1 1 10\n" \
            + "ys2 s2 3 10k\n" \
            + "ys3 s2 s1 1M\n" \
            + "x1 1=1 2=2 n3=3 n12=4 name=BLocK1"

    block_str = block1 + "\n\t" + elems1 + "\n" + block2 + "\n\n" + elems2

    block_repr1 = block1 + "\n" + block2 + "\n" + elems1 + "\n" + elems2
    block_repr2 = block2 + "\n" + block1 + "\n" + elems1 + "\n" + elems2

    block_defs = block_str.split('\n')
    elem = Element(definition='yN1 4 new_node 324k')
    elem_duplicate = Element(definition='y56 2 3 43k')


    # Test 1: Instantiation
    flatten_block = Block('test', ('1', 'n2', 'node3'), block_defs)
    block = Block('test', ('n1', 'n2', 'node3'), block_defs)

    # Test 2: Parsing correctness
    assert len(block.blocks) == 2, 'Incorrect number of blocks detected.'
    assert len(block.elements) == 6, 'Block elements not fully populated.'
    b1 = block.blocks.get('block1')
    b2 = block.blocks.get('block2')
    b1_instance = block.elements[block.elements.index('x1')]
    assert b1, 'Nested block key failure.'
    assert b2, 'Nested block key failure.'
    assert b1.name == 'block1', "Nested block name parsing failed."
    assert b2.name == 'block2', "Nested block name parsing failed."
    assert b1.definition == block1_def.lower(), "Nested block def generation failed."
    assert b2.definition == block2_def.lower(), "Nested block def generation failed."
    assert str(b1).strip() == block1.lower(), 'Nested block to string conv failed.'
    assert b1_instance.block.name == 'block1', "Block instance failed."
    assert block.definition == block_repr1.lower() or \
           block.definition == block_repr2.lower(),\
           'Top level block-string conv. failed.'

    # Test 3: Block manipulation
    es3 = block.element('yS3')
    assert es3 is not None and es3.name == 'ys3', 'Element search failed.'
    block.add(elem)
    assert elem in block.elements, 'Element addition failed.'
    assert elem.name in block.elements, 'Element membership by name failed.'
    assert elem in block.graph[elem.nodes[0]], 'Element addition failed.'
    assert elem in block.graph[elem.nodes[1]], 'Element addition failed.'
    try:
        block.add(elem_duplicate)
    except ValueError:
        pass
    block.remove(elem)
    assert elem not in block.elements, 'Element removal failed.'
    assert elem not in block.graph[elem.nodes[0]], 'Element removal failed.'
    assert block.graph.get(elem.nodes[1]) is None, 'Element removal failed.'

    block.add_block(block)
    assert block in block.blocks, 'Programmatic block addition failed.'
    block.add(block.instance('xTest', n1='n5', n2='n6', node3='n7'))
    assert 'xtest' in block.elements, 'Programmatic instance addition failed.'
    block.remove_block(block)
    assert block.name not in block.blocks, 'Programmatic block removal failed.'
    assert 'xtest' not in block.elements, 'Programmatic instance removal failed.'

    block.short('s2', 's1')
    assert 's2' not in block.graph, 'Shorted node not removed from block.'
    assert 'ys3' not in block.elements, 'Shorted elements not removed.'
    assert len(block.graph['s1']) == 2, 'Incorrect element union after short.'

    # Test 5: block flattening
    flatten_block.flatten()
    assert 'x1' not in flatten_block.elements, 'Flattened block instance not removed.'
    assert len(flatten_block.blocks) == 0, 'Block defs not removed after flattening.'
    assert 'yblock1x1y1' in flatten_block.elements, 'Block instance not expanded.'
    assert 'yblock1x1y2' in flatten_block.elements, 'Block instance not expanded.'
    assert 'block1x13' in flatten_block.graph, 'Internal block node not flattened.'

    # Test 6: block search
    assert flatten_block.element('y6') == 'y6', 'Single element block retreival failed.'
    assert len(flatten_block.elements_like('y')) == 7, 'Multiple element retreival failed.'


@test
def test_netlist_class():
    """Test Netlist class for reading/parsing"""

    # Set up
    net = (".model sw sw0\n"
           ".subckt blah 1 2 3\n"
           "r1 1 2 10.0\n"
           "c1 1 3 12.0\n"
           ".ends blah\n"
           "C1 0 T1 1000.0\n"
           "R1 T1 N001 1000.0\n"
           "G1 N001 0 T1 0 5.0\n"
           ".ic 0 15s 0 1000.0 uic V(T1)=10V\n"
           ".end")
    net_list = ['* Netlist: test'] + net.lower().split('\n')
    definition = net_list[:9] + net_list[10:]
    tfile = open('test.net', 'w')
    tfile.write(net)
    tfile.close()

    # Test 1: Instantiation/ reading netlist file
    ninstance1 = Netlist('test', netlist=net_list[1:])
    ninstance2 = Netlist('test', path="test.net")

    # Test 2: Parsing netlist
    assert 'subckt' not in ninstance1.directives, 'Non-directives not ignored.'
    assert ninstance1.definition == '\n'.join(definition), 'Incorrect definition.'
    assert str(ninstance1) == '\n'.join(net_list), 'Netlist to str failed.'
    assert str(ninstance2) == '\n'.join(net_list), 'Netlist to str failed.'

    # Finalizing
    os.remove('test.net')


#@test
def test_simulator_class():
    """Test circuit simulator"""

    # Set up
    net = ('*Test Circuit',
           '.subckt block in out',
           'r1 in out 1e3',
           'r2 in mid 5e2',
           'c1 mid out 1e-6',
           '.ends block',
           'C1 n1 0 1e-6',
           'R1 n2 n3 1e3',
           'xinstance in=n3 out=0 name=block',
           's1 n1 n2 n1 0 switch',
           '.ic V(n1)=10',
           '.model sw switch von=6 voff=5 ron=1 roff=1e6',
           '.model ekv xsistor type=n',
           '.model diode dd',
           '.end')
    ninstance = Netlist('Test', netlist=net)

    def state_mux(state, action, netlist):
        if state == 1:
            netlist.short('n2', 'n3')   # n2 replaced by n3, r1 deleted
        elif state == 2:
            cap = netlist.element('s1') # reversing state = 1 changes
            cap.nodes[1] = Node('n2')
            netlist.add(Resistor(definition='R1 n2 n3 1e3'))
            netlist.add(Diode(definition='d1 n3 0 dd'))
        elif state == 3:                # adding an NMOS device
            netlist.add(Transistor(definition='M1 n3 n1 0 0 xsistor w=2e-4 l=5e-5'))
            netlist.element('d1').param('area', 1e-1)
        elif state == 4:                # removing NMOS device
            netlist.remove('m1')
            netlist.remove('d1')
        return netlist

    # Test 1: Instantiation and preprocessing
    sim = Simulator(env=ninstance, timestep=1e-6, state_mux=state_mux)
    assert sim.ic == {'v(n1)':'10'}, 'Initial conditions incorrectly parsed.'

    # Test 2: Running simulation
    res1 = sim.run(duration=1e-3)
    res2 = sim.run(duration=1e-3)
    assert 'v(n1)' in res1, 'Incorrect keys in simulation result.'
    assert res1['v(n1)'] > res2['v(n1)'], 'Simulator state does not persist.'

    # Test 3: Switching states and running simulation
    # TODO: Check for correct Netlist -> ahkab.Circuit conversion
    sim.set_state(1, None)
    assert len([e for e in sim.circuit if e.part_id == 'r1']) == 0, \
        "Element deletion not propagated to ahkab circuit."
    sim.run(duration=1e-3)
    sim.set_state(2, None)
    assert len([e for e in sim.circuit if e.part_id == 'r1']) == 1, \
        "Element creation not propagated to ahkab circuit."
    assert len([e for e in sim.circuit if e.part_id == 'd1']) == 1, \
        "Element creation not propagated to ahkab circuit."
    sim.run(duration=1e-3)
    sim.set_state(3, None)
    assert len([e for e in sim.circuit if e.part_id == 'm1']) == 1, \
        "Element creation not propagated to ahkab circuit."
    diodes = [e for e in sim.circuit if e.part_id == 'd1']
    assert diodes[0].device.AREA == 1e-1, "Element attributes not updated."
    sim.run(duration=1e-3)
    sim.set_state(4, None)
    assert len([e for e in sim.circuit if e.part_id == 'm1']) == 0, \
        "Element deletion not propagated to ahkab circuit."
    assert len([e for e in sim.circuit if e.part_id == 'd1']) == 0, \
        "Element deletion not propagated to ahkab circuit."
    sim.run(duration=1e-3)




if __name__ == '__main__':
    print()
    test_flag_generator()
    test_node_class()
    test_element_class()
    test_element_mux()
    test_directive_class()
    test_block_class()
    test_netlist_class()
    test_simulator_class()
    print('\n==========\n')
    print('Tests passed:\t' + str(TESTS_PASSED))
    print('Total tests:\t' + str(NUM_TESTS))
    print()
