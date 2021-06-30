
from collections import defaultdict, namedtuple
from itertools import permutations, product, permutations
from functools import partial
from dataclasses import dataclass


# TODO add repr
Component = namedtuple('Component', ['kind', 'space', 'time'])
Relation = namedtuple('Relation', ['kind', 'first', 'second'])
Structure = namedtuple('Structure', ['relations'])
RelationConstraint = namedtuple('RelationConstraint', ['kind', 'first', 'second'])
StructureClass = namedtuple('StructureClass', ['variables', 'constraints'])
# TODO tbc


def set_first(s):
    for e in s:
        return e
    return None


# FUTURE naming of variables
def print_unities(unities_dict):
    special_chars = {
        'dot': '·',
        'full': '●',
        'empty': '○',
        'dotted': '◌',
    }

    if not any(unities for unities in unities_dict.values()):
        print("Nothing to draw.")
        return
    
    grid = defaultdict(lambda: defaultdict(list))

    for char, unities in unities_dict.items():
        for unity in unities:
            for x, y in unity.space:
                grid[y][x] = special_chars.get(char, char)

    min_y = min(grid.keys())
    max_y = max(grid.keys())

    min_xs, max_xs = zip(*((min(r), max(r)) for r in grid.values()))
    min_x = min(min_xs)
    max_x = max(max_xs)
            
    min_pos = str((min_x, min_y))
    max_pos = str((max_x, max_y))
    hr = '━' * (max_x - min_x + 1)
    
    print(f"{min_pos}")
    print(f"┏{hr}┓")

    for y in range(min_y, max_y + 1):
        print('┃', end='')
        for x in range(min_x, max_x + 1):
            cell = grid[y][x]
            if not cell:
                cell = ' '
            print(cell, end='')
        print('┃')
    
    print(f"┗{hr}┛")
    print(f"{max_pos:>{len(hr)+2}}")


def print_structure_class_dot(clazz):
    # sfdp -Gpad=.5 -Gnodesep=.25 -Tpng -o /tmp/graph.png
    # print('digraph G { dpi="100"; size="30,20"; rankdir=LR; overlap=scale; spline=true; ', end='')
    print('digraph G { ', end='')
    for c in clazz.constraints:
        if c.kind.endswith('-of'):
            color = 'green'
        elif c.kind == 'alive-link':
            color = 'red'
        else:
            color = 'blue'
        print(f'{c.first} -> {c.second} [label="{c.kind}", fontcolor="{color}", color={color}]; ', end='')
    print(' }')
    
    
class AutopoieticAnalysis:

    def __init__(self, observer):
        self.observer = observer

        
    def recognise_components(self, kind, time):
        # FIXME observer should be decoupled here (add func arg?)
        l, t, w, h = self.observer.universe.rects[time]
        for y in range(t, t + h):
            for x in range(l, l + w):
                space = frozenset([(x, y)])
                self.observer.recognise_component(kind, space, time)


    def recognise_relations(self, kind, time):
        components = set()
        for comps in self.observer.components.values():
            components = components.union(comps)
        pairs = permutations(components, 2)
        for first, second in pairs:
            if first.kind == 'dead' and second.kind == 'dead':
                # optimisation: not interested in those relations
                continue
            # if first.kind == 'alive' and second.kind == 'alive':
            #     # optimisation: don't find equivalent links twice
            #     if hash(first) > hash(second):
            #         continue
            self.observer.recognise_relation(kind, first, second)


    # FIXME only glider for now! future: auto-detect
    def get_complex_structures(self, kind, time=None):
        
        # find time or time interval
        if time is None:
            times = range(len(self.observer.universe.states))
        elif isinstance(time, list):
            times = time
        else:
            times = [time]

        # collect info
        unities = set()
        for time_ in times:
            # FIXME observer should be decoupled here (add func arg?)
            l, t, w, h = self.observer.universe.rects[time_]
            for y in range(t, t + h - 5 + 1):
                for x in range(l, l + w - 5 + 1):
                    # FIXME temporary restriction of loop
#                    if -5 <= x <= -3 and 2 <= y <= 4: pass
#                    else: continue
                    #print(f"now considering: ({x}, {y})")
                    space = set(product(range(x, x + 5), range(y, y + 5)))
                    # TODO add rotated biting away corners
                    # TODO add second glider form
                    space.remove((x + 0, y + 0))
                    space.remove((x + 0, y + 1))
                    space.remove((x + 4, y + 0))
                    unity = Unity(frozenset(space), time_)
                    if self.observer.prop('glider', unity):
                        unities.add(unity)

        return unities


@dataclass(frozen=True)
class Unity:

    space: frozenset
    time: int
    
#    def __init__(self, space, time):
#        self.space = set(space)
#        self.time = time
        
    def __iter__(self):
        return iter((self.space, self.time))
    
    def __contains__(self, other):
        if self.time != other.time:
            return False
#        return all(p in self.space for p in other.space)
        return self.space.issuperset(other.space)

    def __repr__(self):
        return f"<U:{self.space}@{self.time}>"

    def __eq__(self, other):
        return self.space == other.space and self.time == other.time

    def copy(self):
        return Unity(self.space, self.time)

    # TODO don't transform unity, but constraints
    def translate(self, delta, time_delta=0):
        dx, dy = delta
        if dx != 0 or dy != 0:
            space = frozenset((x + dx, y + dy) for (x, y) in self.space)
        else:
            space = self.space
        return Unity(space, self.time + time_delta)

    # TODO don't transform unity, but constraints
    def mirror(self, xaxis=True):
        raise NotImplementedError("Not there yet")        

    # TODO don't transform unity, but constraints
    def rotate(self, quartercircles):
        if quartercircles % 4 == 0:
            return self.copy()
        # tricky case: (max_x - min_x) % 2 != (max_y - min_y) % 2
        # no non-ambiguous center
        raise NotImplementedError("Not there yet")
    

class Observer:

    def __init__(self, universe):
        self.universe = universe
        
        # "memory"
        self.components = defaultdict(list)
        self.relations = defaultdict(list)
        self.structures = defaultdict(list)
        self.processes = defaultdict(list)
        self.process_relations = defaultdict(list)
        self.organisations = defaultdict(list)

        # for convenience
        self.component_recognisers = {
            'alive': self._is_component_alive,
            'dead': lambda *args: not self._is_component_alive(*args),
            # 'glider': self._is_component_glider,
        }
        self.relation_recognisers = {
            'dead-alive-boundary': self._recognise_dead_alive_boundary,
            'alive-link': self._recognise_alive_link,
        }
        self._create_spatial_relation_recognisers()

        # as theorised
        # self.properties = {
        #    'alive': self._prop_alive,
        #    'dead': lambda u: not self._prop_alive(u),
        #    'glider': self._prop_glider
        # }
        self.structure_classes = {
            # ([1], []),
            'block': self._make_structure([
                '....',
                '.##.',
                '.##.',
                '....'
            ]),
            'glider1': self._make_structure([
                ' ... ',
                ' .#..',
                '...#.',
                '.###.',
                '.....'
            ])
        }
        self.process_classes = [
            
        ]
        self.organisation_classes = [
            
        ]
        # from pprint import pprint
        # pprint(self.structure_classes)
        # print_structure_class_dot(self.structure_classes['block'])
        # input()


    # move to utility class
    def _make_structure(self, lines):
        grid = [[c for c in line] for line in lines]
        w, h = len(lines[0]), len(lines)
        pos_center = (w // 2, h // 2)
        
        def cdist(pos):
            return (pos_center[0] - pos[0])**2 + (pos_center[1] - pos[1])**2
        pos_all = sorted(product(list(range(w)), list(range(h))), key=cdist)
        
        deltas = list(product((-1, 0, 1), (-1, 0, 1)))
        deltas.remove((0, 0))
        # copied from below :s
        geography = {
            # tower of king
            'west-of': (-1, 0),
            'east-of': (+1, 0),
            'north-of': (0, -1),
            'south-of': (0, +1),
            # bishop of king
            'north-west-of': (-1, -1),
            'north-east-of': (+1, -1),
            'south-west-of': (-1, +1),
            'south-east-of': (+1, +1)
        }
        reverse_geography = {d: n for (n, d) in geography.items()}

        variables = defaultdict()
        variables.default_factory = lambda: len(variables)
        constraints = []

        for x1, y1 in pos_all:
            for delta in deltas:
                dx, dy = delta
                x2, y2 = x1 - dx, y1 - dy
                if x2 < 0 or y2 < 0 or x2 >= w or y2 >= h:
                    continue
                c1, c2 = grid[y1][x1], grid[y2][x2]
                if c1 == ' ' or c2 == ' ':
                    continue
                # print((x1, y2), (x2, y2), (c1, c2))
                index1, index2 = variables[(x1, y1)], variables[(x2, y2)]
                if c1 == '#' and c2 == '#':
                    if ('alive-link', index2, index1) in constraints:
                        continue
                    kinds = ('alive-link', reverse_geography[delta])
                    for kind in kinds:
                        constraint = RelationConstraint(kind, index1, index2)
                        constraints.append(constraint)
                if c1 == '.' and c2 == '#':
                    kinds = ('dead-alive-boundary', reverse_geography[delta])
                    for kind in kinds:
                        constraint = RelationConstraint(kind, index1, index2)
                        constraints.append(constraint)
                    grid[y1][x1] = ' '

        return StructureClass(list(range(len(variables))), constraints)
        

    # deprecated
    # TODO move to some utility class
    def _make_structure_old(self, lines):
        cells = {}
        edges = []
        for y, line in enumerate(lines):
            for x, (cell1, cell2) in enumerate(zip(line[:-1], line[1:])):
                if cell1 == ' ' or cell2 == ' ':
                    continue
                pos1, pos2 = (x, y), (x + 1, y)
                for pos in (pos1, pos2):
                    if pos not in cells:
                        cells[pos] = len(cells)
                edges.append(('west-of', cells[pos1], cells[pos2]))
        transposed = [''.join(r) for r in zip(*lines)]
        for x, line in enumerate(transposed):
            for y, (cell1, cell2) in enumerate(zip(line[:-1], line[1:])):
                if cell1 == ' ' or cell2 == ' ':
                    continue
                pos1, pos2 = (x, y), (x + 1, y)
                for pos in (pos1, pos2):
                    if pos not in cells:
                        cells[pos] = len(cells)
                edges.append(('north-of', cells[pos1], cells[pos2]))
                break
        # return (list(range(len(cells))), edges)
        edges = [(k, cells[p1], cells[p2]) for (k, p1, p2) in edges]
        return StructureClass(list(range(len(cells))), edges)


    # FIXME currently only finds first
    def find_structures(self, kind, time):
        try:
            clazz = self.structure_classes[kind]
        except KeyError:
            raise ValueError("Invalid structure class specified: " + kind)

        # sort most constraining first
        constraint_groups = defaultdict(list)
        for constraint in clazz.constraints:
            constraint_groups[constraint.kind].append(constraint)
        def constraint_key(constraint):
            kind = constraint.kind
            return (len(self.relations[kind]), len(constraint_groups[kind]))
        constraints = sorted(clazz.constraints, key=constraint_key)

        rel_indices = [0] * len(clazz.constraints)
        solution_stack = [({}, [])]
        count = len(clazz.constraints)

        # from pprint import pprint
        # print("XXXXX", kind, time)

        # pprint({k: len(v) for (k, v) in self.relations.items()})
        # pprint(constraints)
        # input()

        # until solution found or maximally backtracked
        while solution_stack and len(solution_stack) <= count:
            constraint_index = len(solution_stack) - 1
            next_rel_index = rel_indices[constraint_index]
            constraint = constraints[constraint_index]
            rel_rest = self.relations[constraint.kind][next_rel_index:]
            cur_variables, cur_relations = solution_stack[-1]

            # if cur_relations:
                # x, y = set_first(cur_relations[0].first.space)
                # if x > -10 and constraint.kind == 'alive-link':
                    # print("CONSTRAINT", constraint_index, constraint)
                    # print(" ", cur_variables)
                    # pprint(cur_relations)
                    # pprint([v for (v, r) in solution_stack])

            # ss = [0, 5, 7, 8]
            # pp = [s for s in ss if s in cur_variables]
            # t = len(pp)
            # y = len([1 for p in pp if cur_variables[p].kind == 'alive' and set_first(cur_variables[p].space)[0] > -24])
            # m = "***" if y >= 3 else ""
            # print("CONSTRAINT", constraint_index, constraint, f"{y}/{t} {m}")
            # pprint(cur_variables)

            comp1 = cur_variables.get(constraint.first)
            comp2 = cur_variables.get(constraint.second)

            for rel_index, rel in enumerate(rel_rest, start=next_rel_index):

                if rel.first.time != time:
                    continue
                
                if comp1 and rel.first != comp1:
                    continue
                
                if comp2 and rel.second != comp2:
                    continue

                if rel in cur_relations:
                    continue

                if not comp1 and rel.first in cur_variables.values():
                    continue

                if not comp2 and rel.second in cur_variables.values():
                    continue

                # constraint satisfied by relation! store.
                variables = cur_variables.copy()
                if constraint.first not in variables:
                    variables[constraint.first] = rel.first
                if constraint.second not in variables:
                    variables[constraint.second] = rel.second
                    
                relations = cur_relations.copy()
                relations.append(rel)
                solution = (variables, relations)
                solution_stack.append(solution)

                rel_indices[constraint_index] = rel_index + 1
                break

            else:
                # loop ended not via break
                # i.e. no relations left to check
                # i.e. constraint not satisfied
                # therefore backtrack
                solution_stack.pop()
                # next time we're at this index, start over
                rel_indices[constraint_index] = 0

        if len(solution_stack) <= count:
            # print(f"+++ NO {kind} FOUND at {time} +++")
            return []

        # print(f"+++ FOUND {kind} at {time} +++")

        structure = Structure(relations)
        self.structures[kind].append(structure)
        return [structure]
        

    def _is_component_alive(self, space, time):
        if len(space) > 1:
            raise ValueError("Multi-cell alive-query not supported.")
        alive = (set_first(space) in self.universe.states[time])
        return alive


    # unused atm
    def _is_component_glider(self, space, time):

        if len(space) != 22:
            return False

        #alive = [u for u in self.unities['alive'] if u.space.issubset(space)]
        alive = [u for u in self.unities['alive'] if u in unity and u.time == time]

        if len(alive) != 5:
            return False
        
        #dead = [u for u in self.unities['dead'] if u.space.issubset(space)]
        dead = [u for u in self.unities['dead'] if u in unity and u.time == time]
        
        for perm in permutations(alive):
            u1, u2, u3, u4, u5 = perm
            if not self.relation('north-west-of', u1, u2): continue
            if not self.relation('north-of', u2, u3): continue
            if not self.relation('east-of', u3, u4): continue
            if not self.relation('east-of', u4, u5): continue
            # success! alive unities in right constellation

            # now let's check for some dead ones in right spots
            if not any(self.relation('south-of', u1, ud) for ud in dead): continue
            if not any(self.relation('north-west-of', u3, ud) for ud in dead): continue
            if not any(self.relation('north-east-of', u5, ud) for ud in dead): continue
            # success! alive unities in middle of pattern
            
            # now let's check that the right cells are not specified
            if any(self.relation('south-west-west-of', u1, ud) for ud in dead): continue
            if any(self.relation('south-east-east-of', u1, ud) for ud in dead): continue
            if any(self.relation('south-south-east-of', u5, ud) for ud in dead): continue
            # success! we have a glider in .:, orientation
            
            return True
        
        # no permutation of suitably arranged alive cells could be found
        return False

        
    def recognise_component(self, kind, space, time):
        try:
            recogniser = self.component_recognisers[kind]
        except KeyError:
            ValueError("Invalid compnent kind specified: " + kind)
        if recogniser(space, time):
            component = Component(kind, space, time)
            self.components[kind].append(component)
            return component
        return None

    
    def _create_spatial_relation_recognisers(self): 
        geography = {
            # tower of king
            'west-of': (-1, 0),
            'east-of': (+1, 0),
            'north-of': (0, -1),
            'south-of': (0, +1),
            # bishop of king
            'north-west-of': (-1, -1),
            'north-east-of': (+1, -1),
            'south-west-of': (-1, +1),
            'south-east-of': (+1, +1),
            # knight
            # 'north-north-west-of': (-1, +2),
            # 'north-north-east-of': (+1, +2),
            # 'north-west-west-of': (-2, +1),
            # 'north-east-east-of': (+2, +1),
            # 'south-west-west-of': (-2, -1),
            # 'south-east-east-of': (+2, -1),
            # 'south-south-west-of': (-1, -2),
            # 'south-south-east-of': (+1, -2),
        }
        for kind, delta in geography.items():
            recogniser = partial(self._recognise_spatial_relation, delta)
            self.relation_recognisers[kind] = recogniser

            
    def _recognise_spatial_relation(self, delta, comp1, comp2):
        if len(comp1.space) != 1 or len(comp2.space) != 1:
            raise ValueError("Non atomic space provided!")
        if comp1.time != comp2.time:
            return False
        x1, y1 = set_first(comp1.space)
        x2, y2 = set_first(comp2.space)
        dx, dy = delta
        return x1 - dx == x2 and y1 - dy == y2


    def _recognise_dead_alive_boundary(self, comp1, comp2):
        if len(comp1.space) != 1 or len(comp2.space) != 1:
            raise ValueError("Non atomic space provided!")
        if comp1.time != comp2.time:
            return False
        x1, y1 = set_first(comp1.space)
        x2, y2 = set_first(comp2.space)
        if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
            return False
        return comp1.kind == 'dead' and comp2.kind == 'alive'


    def _recognise_alive_link(self, comp1, comp2):
        if len(comp1.space) != 1 or len(comp2.space) != 1:
            raise ValueError("Non atomic space provided!")
        if comp1.time != comp2.time:
            return False
        x1, y1 = set_first(comp1.space)
        x2, y2 = set_first(comp2.space)
        if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
            return False
        return comp1.kind == 'alive' and comp2.kind == 'alive'
                


    def recognise_relation(self, kind, comp1, comp2):
        try:
            recogniser = self.relation_recognisers[kind]
        except KeyError:
            ValueError("Invalid relation kind specified: " + kind)
        if recogniser(comp1, comp2):
            relation = Relation(kind, comp1, comp2)
            self.relations[kind].append(relation)
            return relation
        return None


    # unused atm
    def _process_glider_travel_phase4(self, unities_in, unities_out):
        time1 = set_first(unities_in).time
        time2 = set_first(unities_out).time
        tdelta = time2 - time1

        if not all(u.time == time1 for u in unities_in):
            return False
        if not all(u.time == time2 for u in unities_out):
            return False

        for delta in product([-1, 1], [-1, 1]):
            ui_translated = frozenset(u.translate(delta, tdelta) for u in unities_in)
            if ui_translated == unities_out:
                return True

        return False
        

    # TODO redo
    # TODO review: double-set of unities vs: two unities?
    def process(self, kind, unities_in, unities_out):

        if kind == 'glider-travel-4':
            return self._process_glider_travel_phase4(unities_in, unities_out)
        
        raise ValueError("Invalid kind specified: " + kind)
    
