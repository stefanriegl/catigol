
# Implementation note:
# This file uses type hints that require Python version >=3.9.
#   https://docs.python.org/3/library/typing.html
# If you get an error message such as
#   TypeError: 'type' object is not subscriptable
# then make sure to run this script (or Golly) with a suitable Python version.


from collections import defaultdict
from typing import NamedTuple
from itertools import permutations, product, permutations
from functools import partial
from dataclasses import dataclass


import util
from util import set_first

from importlib import reload
import util
reload(util)


class Component(NamedTuple):
    """DOC"""
    kind: str
    space: frozenset[tuple[float]]
    time: float
    # TODO decide: add field "structure: Structure" here?
    def __repr__(self) -> str:
        return f'<C {self.kind} l{len(self.space)} t{self.time}>'

class ComponentRelation(NamedTuple):
    """DOC"""
    kind: str
    first: Component
    second: Component
    def __repr__(self) -> str:
        return f'<CR {self.kind}\n  {self.first}\n  {self.second}>'

class Structure(NamedTuple):
    """DOC"""
    relations: frozenset[ComponentRelation]
    # FUTURE have this as field instead of function?
    def components(self) -> frozenset[Component]:
        comps = {r.first for r in self.relations}
        comps.update({r.second for r in self.relations})
        return frozenset(comps)
    def __repr__(self) -> str:
        comps = self.components()
        tt = set_first(self.relations).first.time if self.relations else '*'
        return f'<S c{len(comps)} r{len(self.relations)} t{tt}>'



# TODO decide: could Process be conflated with [Component]Relation?
# maybe yes theoretically, but it won't be practical or easy to follow
# relation: at one time. process: multiple start/end structures
class Process(NamedTuple):
    """DOC"""
    start: frozenset[Structure]
    end: frozenset[Structure]
    def __repr__(self) -> str:
        return f'<P {self.start} {self.end}>'

class ProcessRelation(NamedTuple):
    """DOC"""
    kind: str
    first: Process
    second: Process
    def __repr__(self) -> str:
        return f'<PR {self.kind}\n  {self.first}\n  {self.second}>'

class Organisation(NamedTuple):
    """DOC"""
    relations: frozenset[ProcessRelation]
    def __repr__(self) -> str:
        return f'<O r{len(self.relations)}>'

    

class ComponentRelationConstraint(NamedTuple):
    """DOC"""
    kind: str
    first: Component
    second: Component
    def __repr__(self) -> str:
        return f'<CRC {self.kind}\n  {self.first}\n  {self.second}>'

class StructureClass(NamedTuple):
    """DOC"""
    # FUTURE variables variable could be optimised
    variables: list[int]
    constraints: frozenset[ComponentRelationConstraint]
    def __repr__(self) -> str:
        return f'<SC v{len(self.variables)} c{len(self.constraints)}>'


    
class ProcessRelationConstraint(NamedTuple):
    """DOC"""
    kind: str
    first: Process
    second: Process
    def __repr__(self) -> str:
        return f'<PRC {self.kind}\n  {self.first}\n  {self.second}>'

class OrganisationClass(NamedTuple):
    """DOC"""
    # FUTURE variables variable could be optimised
    variables: list[int]
    constraints: frozenset[ProcessRelationConstraint]
    def __repr__(self) -> str:
        return f'<OC v{len(self.variables)} c{len(self.constraints)}>'

    
    
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
        component_lists = self.observer.components.values()
        components = set(comp for cl in component_lists for comp in cl)
        # FIXME this can explode
        pairs = permutations(components, 2)
        # pairs = permutations(set(self.observer.components.values()), 2)
        for first, second in pairs:
            # ignore pairs of spaces without overlap (boundaries count)
            if first.kind == 'dead' and second.kind == 'dead':
                # optimisation: not interested in those relations
                continue
            # if first.kind == 'alive' and second.kind == 'alive':
            #     # optimisation: don't find equivalent links twice
            #     if hash(first) > hash(second):
            #         continue
            self.observer.recognise_relation(kind, first, second)


class Observer:

    def __init__(self, universe):
        self.universe = universe
        
        # "memory"
        self.components = defaultdict(list)
        # TODO rename to component_relations
        self.relations = defaultdict(list)
        self.structures = defaultdict(list)
        self.processes = defaultdict(list)
        self.process_relations = defaultdict(list)
        self.organisations = defaultdict(list)

        self.component_recognisers = {
            'alive': self._is_component_alive,
            'dead': lambda *args: not self._is_component_alive(*args),
            'glider': self._is_component_glider,
        }
        self.relation_recognisers = {
            'dead-alive-boundary': self._recognise_dead_alive_boundary,
            'alive-link': self._recognise_alive_link,
        }
        self._create_spatial_relation_recognisers()

        self.structure_classes = {
            # ([1], []),
            'block': self._make_structure_class([
                '....',
                '.##.',
                '.##.',
                '....'
            ]),
        }
        
        glider_patterns = {
            'w': [
                ' ... ',
                ' .#..',
                '...#.',
                '.###.',
                '.....'
            ],
            'r': [
                ' ... ',
                ' .#..',
                '..##.',
                '.#.#.',
                '.....'
            ]
        }
        rotations = {
            'se': lambda x: x,
            'sw': util.pattern_rotate90,
            'nw': util.pattern_rotate180,
            'ne': util.pattern_rotate270,
        }
        chiralities = {
            '1': lambda x: x,
            '2': util.pattern_transpose,
        }
        for kgp, glider_pattern in glider_patterns.items():
            for kr, fn_rot in rotations.items():
                for kc, fn_chi in chiralities.items():
                    pattern = fn_chi(fn_rot(glider_pattern))
                    clazz = self._make_structure_class(pattern)
                    key = f'glider-{kr}-{kgp}{kc}'
                    self.structure_classes[key] = clazz

        self.organisation_classes = [
            
        ]


    # move to utility class
    def _make_structure_class(self, lines):
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
                        constraint = ComponentRelationConstraint(kind, index1, index2)
                        constraints.append(constraint)
                if c1 == '.' and c2 == '#':
                    kinds = ('dead-alive-boundary', reverse_geography[delta])
                    for kind in kinds:
                        constraint = ComponentRelationConstraint(kind, index1, index2)
                        constraints.append(constraint)
                    grid[y1][x1] = ' '

        return StructureClass(list(range(len(variables))), constraints)
        

    # deprecated, unused
    # TODO move to some utility class
    def _make_structure_classes_old(self, lines):
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


    # FUTURE should be automated. any structure could be component
    # for now this only supports "remembering" structures,
    # but not detecting by analysing a random space-time
    def _is_component_glider(self, space, time):
        for structure in self.structures.get('glider', []):
            components = structure.components()
            if not components or set_first(components).time != time:
                continue
            structure_space = set()
            structure_space.update(c.space for c in components)
            if structure_space == space:
                return True
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
            relation = ComponentRelation(kind, comp1, comp2)
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
    
