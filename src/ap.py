
# Implementation note:
# This file uses type hints that require Python version >=3.9.
#   https://docs.python.org/3/library/typing.html
# If you get an error message such as
#   TypeError: 'type' object is not subscriptable
# then make sure to run this script (or Golly) with a suitable Python version.


from collections import defaultdict
from typing import NamedTuple, Dict, Tuple, List, Set
from itertools import product
#from dataclasses import dataclass

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

    @classmethod
    def parse(clazz, lines):
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


# class SpaceTime(NamedTuple):
#     space: list[tuple[int]]
#     time: int
#     # TODO repr
    

# class Environment:
#     """To be subclassed."""
#     def __init__(self):
#         pass
#     def get_at(self, space, time):
#         pass
    
    
class Observer:

    def __init__(self):
        # self.environment = environment
        # Note: The `kind` keys are help for development and debugging. They
        # are not meaningful in the scope of the model.

        # TODO create custom types, not this nested mess
        # TODO rename to unities
        self.components: Dict[int, Dict[str, Component]] = defaultdict(lambda: defaultdict(list))
        # TODO rename to component_relations
        self.relations: Dict[int, Dict[str, ComponentRelation]] = defaultdict(lambda: defaultdict(list))

        self.structures: Dict[int, Dict[str, Structure]] = defaultdict(lambda: defaultdict(list))
        self.processes: Dict[int, Process] = defaultdict(list)
        self.process_relations: Dict[int, ProcessRelation] = defaultdict(list)
        self.organisations: Dict[int, Organisation] = defaultdict(list)

        self.component_recognisers = {}
        self.relation_recognisers = {}
        self.process_recognisers = {}


    def recognise_component(self, kind, space, time):
        """Aka. distinction. 
        `kind` is a property name.
        `space` is a index list indicating a sub-space."""
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
    
            
