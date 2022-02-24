
# Implementation note:
# This file uses type hints that require Python version >=3.9.
#   https://docs.python.org/3/library/typing.html
# If you get an error message such as
#   TypeError: 'type' object is not subscriptable
# then make sure to run this script (or Golly) with a suitable Python version.


from collections import defaultdict
from typing import NamedTuple
from itertools import permutations, permutations, product
from functools import partial
from dataclasses import dataclass

from importlib import reload
import util
reload(util)
from util import set_first #, make_structure_class


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
    kind: str
    start: frozenset[Component]
    end: frozenset[Component]
    def __repr__(self) -> str:
        return f'<P {self.kind} {self.start} {self.end}>'

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

    
class Observer:

    def __init__(self):        
        # "memory"
        # TODO rename to unities
        self.components = defaultdict(lambda: defaultdict(list))
        # TODO rename to component_relations
        self.relations = defaultdict(lambda: defaultdict(list))
        self.structures = defaultdict(lambda: defaultdict(list))
        self.processes = defaultdict(list)
        self.process_relations = defaultdict(list)
        self.organisations = defaultdict(list)

        self.component_recognisers = {}
        self.relation_recognisers = {}
        self.process_recognisers = {}
            
    def recognise_component(self, kind, space, time):
        try:
            recogniser = self.component_recognisers[kind]
        except KeyError:
            raise ValueError("Invalid compnent kind specified: " + kind)
        if recogniser(space, time):
            component = Component(kind, space, time)
            self.components[time][kind].append(component)
            return component
        return None

    def recognise_relation(self, kind, comp1, comp2):
        try:
            recogniser = self.relation_recognisers[kind]
        except KeyError:
            raise ValueError("Invalid relation kind specified: " + kind)
        if recogniser(comp1, comp2):
            relation = ComponentRelation(kind, comp1, comp2)
            self.relations[comp1.time][kind].append(relation)
            return relation
        return None

    def recognise_process(self, kind, comps_start, comps_end):
        try:
            recogniser = self.process_recognisers[kind]
        except KeyError:
            raise ValueError("Invalid process kind specified: " + kind)
        if recogniser(comps_start, comps_end):
            process = Process(kind, comps_start, comps_end)
            self.processes[kind].append(process)
            return process
        return None

    def get_all_components_at(self, time):
        component_lists = self.components[time].values()
        components = set(comp for cl in component_lists for comp in cl)
        return components
    
            
class GliderObserver(Observer):

    def __init__(self, universe):
        super().__init__()

        self.universe = universe
        
        # component (unity)
        self.component_recognisers = {
            'alive': self._is_component_alive,
            'dead': lambda *args: not self._is_component_alive(*args),
            'glider': self._is_component_glider,
        }

        # component relation
        self.relation_recognisers = {
            'dead-alive-boundary': self._recognise_dead_alive_boundary,
            'alive-link': self._recognise_alive_link,
        }
        self._create_spatial_relation_recognisers()

        # process
        self.process_recognisers = {
            'emptiness': self._recognise_process_emptiness,
            'birth': self._recognise_process_birth,
            'living': self._recognise_process_living,
            'death': self._recognise_process_death,
            'block': self._recognise_process_block,
            'glider': self._recognise_process_glider,
        }
        # self._create_glider_process_recognisers()

        # structure class
        self.structure_classes = {
            'block': make_structure_class([
                '....',
                '.##.',
                '.##.',
                '....'
            ]),
            'blinker-h': make_structure_class([
                '.....',
                '.###.',
                '.....',
            ]),
            'blinker-v': make_structure_class([
                '...',
                '.#.',
                '.#.',
                '.#.',
                '...',
            ]),
        }
        # nbh_sc_c = frozenset([
        #     ComponentRelationConstraint('north-west-of', 0, 4),
        #     ComponentRelationConstraint('north-of',      1, 4),
        #     ComponentRelationConstraint('north-east-of', 2, 4),
        #     ComponentRelationConstraint('west-of',       3, 4),
        #     ComponentRelationConstraint('east-of',       5, 4),
        #     ComponentRelationConstraint('south-west-of', 6, 4),
        #     ComponentRelationConstraint('south-of',      7, 4),
        #     ComponentRelationConstraint('south-est-of',  8, 4),
        # ])
        # nbh_sc = StructureClass(list(range(9)), nbh_sc_c)
        # self.structure_classes['neighbourhood'] = nbh_sc
        self._create_glider_structure_classes()

        # organisation class
        self.organisation_classes = {
        }

        # optimisation
        self.component_structures = {}


    def recognise_all_components(self, kinds=None, times=None):
        if not kinds:
            kinds = self.component_recognisers.keys()
        if not times:
            times = list(range(len(self.universe.rects)))
        for time in times:
            l, t, w, h = self.universe.rects[time]
            for y in range(t, t + h):
                for x in range(l, l + w):
                    space = frozenset([(x, y)])
                    for kind in kinds:
                        self.recognise_component(kind, space, time)

           
    def recognise_all_relations(self, kinds=None, times=None, skip_double_dead=True):
        if not kinds:
            kinds = self.relation_recognisers.keys()
        if not times:
            times = list(range(len(self.universe.rects)))
        for time in times:
            components = self.get_all_components_at(time)
            # FIXME this can explode
            pairs = permutations(components, 2)
            # pairs = permutations(set(self.observer.components.values()), 2)
            for first, second in pairs:
                # TODO ignore pairs of spaces without overlap (boundaries count)

                # optimisation: not interested in empty relations
                if skip_double_dead and first.kind == 'dead' and second.kind == 'dead':
                    continue
                
                # if first.kind == 'alive' and second.kind == 'alive':
                #     # optimisation: don't find equivalent links twice
                #     if hash(first) > hash(second):
                #         continue
                for kind in kinds:
                    self.recognise_relation(kind, first, second)

           
    def recognise_all_processes(self, kinds=None, times_start=None, times_end=None):
        if not kinds:
            kinds = self.process_recognisers.keys()
        if not times_start:
            times_start = list(range(len(self.universe.rects)))
        if not times_end:
            times_end = list(range(len(self.universe.rects)))
        time_frames = [(s, e) for (s, e) in product(times_start, times_end) if s < e]
        for time_start, time_end in time_frames:
            comps_all_start = self.get_all_components_at(time_start)
            comps_all_end = self.get_all_components_at(time_end)
            # only consider 3x3->1x1 (basic GoL rules) grid-like component processes and
            # processes that map from one to one component
            for comp_center_start, comp_center_end in product(comps_all_start, comps_all_end):
                if len(comp_center_start.space) > 1 or len(comp_center_end.space) > 1:
                    if len(comp_center_start.space) > 1 and len(comp_center_end.space) > 1:
                        comps_start = frozenset([comp_center_start])
                        comps_end = frozenset([comp_center_end])
                        for kind in kinds:
                            self.recognise_process(kind, comps_start, comps_end)
                    continue

                # basic GoL rules below
                if comp_center_start.space != comp_center_end.space:
                    continue
                pos1 = set_first(comp_center_start.space)
                comps_start = set()
                for comp_start in comps_all_start:
                    if len(comp_start.space) > 1:
                        continue
                    pos2 = set_first(comp_start.space)
                    if abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1:
                        comps_start.add(comp_start)
                if len(comps_start) != 9:
                    continue
                # skip entire neighbourhoods of uncoupled emptiness
                if sum(1 for c in comps_start if c.kind == 'dead') == 9:
                    continue
                comps_start = frozenset(comps_start)
                comps_end = frozenset([comp_center_end])
                for kind in kinds:
                    self.recognise_process(kind, comps_start, comps_end)
                

    def decompose_structured_process(self, process):
        # assume start and end components are singular components
        if len(process.start) > 1 or len(process.end) > 1:
            return None
        comp_start = set_first(process.start)
        comp_end = set_first(process.end)
        # assume start and end components are structures
        try:
            struct_start = self.component_structures[comp_start]
            struct_end = self.component_structures[comp_end]
        except KeyError:
            return None
        comps_start = struct_start.components()
        comps_end = struct_end.components()
        kind = f'{process.kind}-decomposed'
        decomposed_process = Process(kind, comps_start, comps_end)
        return decomposed_process


    def compose_processes(self, processes):
        comps_start = set(c for p in processes for c in p.start)
        comps_end = set(c for p in processes for c in p.end)
        intersection = comps_start.intersection(comps_end)
        comps_start.difference_update(intersection)
        comps_end.difference_update(intersection)
        kind = f'process-composed'
        composed_process = Process(kind, frozenset(comps_start), frozenset(comps_end))
        return composed_process      
    
                    
    def _create_glider_structure_classes(self):
        glider_patterns = {
            'w': [
                ' ... ',
                ' .#..',
                '...#.',
                '.###.',
                '.....'
            ],
            'r': [
                '.....',
                '.#.#.',
                '..##.',
                ' .#..',
                ' ... '
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
                    pattern = fn_rot(fn_chi(glider_pattern))
                    clazz = make_structure_class(pattern)
                    key = f'glider-{kr}-{kgp}{kc}'
                    self.structure_classes[key] = clazz
                    # print(f'key: {key}')
                    # for line in pattern:
                        # print(f'  {line}')

    
    # FIXME currently only finds first
    # This is a bit dirty. If structures have classes, then those are
    # basically also component classes. That means, when a structure of a
    # certain class (kind) is found, observer can also remember it as a
    # component.
    # A more correct way would be to push identified structures into
    # recognise_component(), but that takes more computational effort than
    # this project warrants. :/
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
            # kind = constraint.kind
            # return (len(self.relations[kind]), len(constraint_groups[kind]))
            len_rels = len(self.relations[time][constraint.kind])
            len_const = len(constraint_groups[kind])
            return (len_rels, len_const)
        constraints = sorted(clazz.constraints, key=constraint_key)

        rel_indices = [0] * len(clazz.constraints)
        solution_stack = [({}, [])]
        count = len(clazz.constraints)

        # from pprint import pprint
        # print("XXXXX", kind, time)
        # print(clazz.constraints)

        # pprint({k: len(v) for (k, v) in self.relations.items()})
        # pprint(constraints)
        # input()

        # until solution found or maximally backtracked
        while solution_stack and len(solution_stack) <= count:
            constraint_index = len(solution_stack) - 1
            next_rel_index = rel_indices[constraint_index]
            constraint = constraints[constraint_index]
            rel_rest = self.relations[time][constraint.kind][next_rel_index:]
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

            # d_v = len(cur_variables)
            # d_c = len(cur_relations)
            # d_va = len(clazz.variables)
            # d_ca = count
            # if d_c >= 0 and kind == 'neighbourhood':
            #     print("CONSTRAINT", f"v{d_v}/{d_va} c{d_c}/{d_ca}")
            #     # print(cur_variables)
            #     # print(cur_relations)
            #     print(self.relations[time][constraint.kind][next_rel_index:])
            #     print(self.relations[time][constraint.kind])
            #     print(self.relations[time])
            # pprint(cur_variables)

            # fall back to None if not found
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
        self.structures[time][kind].append(structure)

        # quick fix: also store as component of same-named kind
        # FIXME this might rather be done via recognise_component somehow
        space = set()
        space = space.union(*[c.space for c in structure.components()])
        component = Component(kind, frozenset(space), time)
        self.components[time][kind].append(component)
        self.component_structures[component] = structure
        
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
        for structure in self.structures[time].get('glider', []):
            components = structure.components()
            # TODO tweak after test
            if not components or set_first(components).time != time:
                continue
            structure_space = set()
            structure_space.update(c.space for c in components)
            if structure_space == space:
                return True
        return False

    

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

    def _get_neighbourhood_center_component(self, comps):
        pos_x = sum(set_first(c.space)[0] for c in comps) // 9
        pos_y = sum(set_first(c.space)[1] for c in comps) // 9
        space = frozenset([(pos_x, pos_y)])
        center_comp = [c for c in comps if c.space == space][0]
        return center_comp

    # deprecated
    def _count_alive_dead(self, comps):
        center_comp = self._get_neighbourhood_center_component(comps)
        count_alive = sum(1 for c in comps if c.kind == 'alive')
        count_dead = sum(1 for c in comps if c.kind == 'dead')
        return (center_comp.kind == 'alive', count_alive, count_dead)

    def _recognise_process_emptiness(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = set_first(comps_end)
        return comp_start.kind == 'dead' and comp_end.kind == 'dead'
            
    def _recognise_process_birth(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = set_first(comps_end)
        return comp_start.kind == 'dead' and comp_end.kind == 'alive'

    def _recognise_process_living(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = set_first(comps_end)
        return comp_start.kind == 'alive' and comp_end.kind == 'alive'
    
    def _recognise_process_death(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = set_first(comps_end)
        return comp_start.kind == 'alive' and comp_end.kind == 'dead'

    
    def _recognise_process_block(self, comps_start, comps_end):
        # we're only supporting single-comp to single-comp processes for now
        if len(comps_start) > 1 or len(comps_end) > 1:
            return False
        comp_start = set_first(comps_start)
        comp_end = set_first(comps_end)
        return comp_start.kind == 'block' and comp_end.kind == 'block'
    
    def _recognise_process_glider(self, comps_start, comps_end):
        # we're only supporting single-comp to single-comp processes for now
        if len(comps_start) > 1 or len(comps_end) > 1:
            return False
        comp_start = set_first(comps_start)
        comp_end = set_first(comps_end)
        if not comp_start.kind.startswith('glider-'):
            return False
        if not comp_end.kind.startswith('glider-'):
            return False
        _, orientation_start, key_start = comp_start.kind.split('-')
        _, orientation_end, key_end = comp_end.kind.split('-')
        if orientation_start != orientation_end:
            return False
        key = key_start + key_end
        intersection = comp_start.space.intersection(comp_end.space)
        return key in 'r1w2r2w1r1' and len(intersection) in (18, 20)
    


def make_structure_class(lines):
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
        # tower & king
        'west-of': (-1, 0),
        'east-of': (+1, 0),
        'north-of': (0, -1),
        'south-of': (0, +1),
        # bishop & king
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
            # TODO this part can use some optimisation
            # - do not have both west-of and east-of relations etc.
            # - requires changing different parts of code too
            # - allows simplification in the then-blocks below
            # print((x1, y2), (x2, y2), (c1, c2))
            index1, index2 = variables[(x1, y1)], variables[(x2, y2)]
            # quick'n'dirty hack to detect component sets for minimal processes
            # if c1 == '?' and c2 == '?':
            #     kind = reverse_geography[delta]
            #     constraint = ComponentRelationConstraint(kind, index1, index2)
            #     constraints.append(constraint)
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

