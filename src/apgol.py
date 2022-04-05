
from typing import NamedTuple
from functools import partial
from itertools import permutations, combinations, product, cycle
from collections import defaultdict, Counter

import ap
import util

# Better integration for golly
import importlib
importlib.reload(ap)
importlib.reload(util)



class Location(NamedTuple):
    """DOC"""
    x: int
    y: int
    def __repr__(self) -> str:
        return '({}|{})'.format(self.x, self.y)


class GolCell:
    
    def __init__(self, location, time, value, ancestor=None, descendant=None):
        self.location = location
        self.time = time
        self.value = value
        # neighbours only at own point in time
        self._neighbours = [None] * 9
        self._neighbours[4] = self  # (dx, dy) == (0, 0)
        self.ancestor = ancestor
        self.descendant = descendant
        
    def _get_neighbour_index(self, dx, dy):
        if abs(dx) > 1 or abs(dy) > 1:
            raise ValueError("Invalid delta: {}, {}".format(dx, dy))
        # return (dx + 1) + 3 * (dy + 1)
        return dx + 3 * dy + 4
    
    def get_neighbour(self, dx, dy):
        return self._neighbours[self._get_neighbour_index(dx, dy)]
    
    def get_neighbours(self):
        return self._neighbours[0:4] + self._neighbours[5:9]
    
    def set_neighbour(self, neighbour, dx, dy):
        """Only to be used during setup."""
        if dx == 0 and dy == 0:
            raise ValueError("Setting self not allowed.") 
        self._neighbours[self._get_neighbour_index(dx, dy)] = neighbour

    # def set_at(self, value, time):
    #     """Only to be used during setup."""
    #     cell = (self if time == self.time else self._timeline[time])
    #     cell.value = value

    # def get_at(self, time):
    #     cell = (self if time == self.time else self._timeline[time])
    #     return cell.value

    def __repr__(self):
        flag = '#' if self.value else '.'
        return '<CELL {}@{} {}>'.format(self.location, self.time, flag)
    

# rather GolWorld or GolHistory or GolEnvHist
class GolEnvironment(ap.Discrete2DEnvironment):

    def __init__(self, golly, rect):
        self._golly = golly
        left, top, width, height = rect
        self.offset = (left, top)
        self.size = (width, height)
        self._history = []

    def setup(self):
        self._golly.reset()
        self._add_history_entry()

    def simulate_step(self):
        self._golly.step()
        self._add_history_entry()
                
    def get_cell(self, location, time):
        ix, iy = location[0] - self.offset[0], location[1] - self.offset[1]
        width, height = self.size
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            raise ValueError("Coordinates out of bounds.")
        return self._history[ix + iy * width + time * width * height]

    # is it needed?
    def get_cells(self, time):
        window = self.size[0] * self.size[1]
        return self._history[time * window:(time + 1) * window]

    def _get_cell_rect(self, pos, size, cells, cells_width):
        px, py = pos
        sx, sy = size
        for iy in range(py, py + sy):
            index = px + py * cells_width
            yield cells[index:index + sx]
    
    def get_subspaces(self, size, time):
        ss_width, ss_height = size
        env_width, env_height = self.size
        cells = self.get_cells(time)
        for iy in range(env_height - ss_height + 1):
            for ix in range(env_width - ss_width + 1):
                rect = self._get_cell_rect((ix, iy), size, cells, env_width)
                yield frozenset(*rect)
    
    # def _golly_get_sel_values(self, rect):
    #     offset_x, offset_y, width, height = rect
    #     values = [False] * (width * height)
    #     cells = self._golly.getcells(rect)
    #     locations = list(zip(cells[0::2], cells[1::2]))
    #     for abs_x, abs_y in locations:
    #         index = (abs_x - offset_x) + (abs_y - offset_y) * width
    #         values[index] = True
    #     return values

    def _add_history_entry(self):
        offset_x, offset_y = self.offset
        width, height = self.size
        rect = [offset_x, offset_y, width, height]
        grid = []

        time = self.get_duration()
        cell_data = self._golly.getcells(rect)
        alive_cells = list(zip(cell_data[0::2], cell_data[1::2]))

        for y in range(offset_y, offset_y + height):
            for x in range(offset_x, offset_x + width):
                #value = values[ix + iy * width]
                #cell = self._grid[x + y * width]
                #location = Location(self.offset[0] + ix, self.offset[1] + iy)
                value = ((x, y) in alive_cells)
                cell = GolCell(Location(x, y), time, value)
                #cell.set_at(value, time)
                grid.append(cell)

        self._history += grid

        # TODO merge this into loop above
        for iy in range(height):
            for ix in range(width):
                cell = grid[ix + iy * width]

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if not dx and not dy:
                            continue
                        jx, jy = ix + dx, iy + dy
                        if jx < 0 or jx >= width or jy < 0 or jy >= height:
                            continue
                        neighbour = grid[jx + jy * width]
                        cell.set_neighbour(neighbour, dx, dy)

                if time > 0:
                    index = ix + iy * width + (time - 1) * width * height
                    ancestor = self._history[index]
                    cell.ancestor = ancestor
                    ancestor.descendant = cell

    def get_duration(self):
        width, height = self.size
        return len(self._history) // (width * height)


class GolObserver(ap.Observer):

    def __init__(self, environment):
        super().__init__()

        self.environment = environment
        
        # component (unity)
        self.component_recognisers = {
            #'alive': self._is_component_alive,
            #'dead': lambda *args: not self._is_component_alive(*args),
            #'glider': self._is_component_glider,
            'alive-single': self._is_component_alive,
            'alive-contingent': self._is_component_alive,
            'alive-bounded': self._is_component_alive_bounded,
            'alive-bounded-env': self._is_component_alive_bounded_env,
        }

        # component relation
        self.relation_recognisers = {
            #'dead-alive-boundary': self._recognise_dead_alive_boundary,
            'alive-single-link': self._recognise_alive_link,
        }
        self._create_spatial_relation_recognisers()

        # process
        self.process_recognisers = {
            # 'emptiness': self._recognise_process_emptiness,
            # 'birth': self._recognise_process_birth,
            # 'living': self._recognise_process_living,
            # 'death': self._recognise_process_death,
            # 'block': self._recognise_process_block,
            # 'glider': self._recognise_process_glider,
            'bounded-transformation': self._recognise_process_bounded_transformation,
        }
        # self._create_glider_process_recognisers()

        # structure class
        self.structure_classes = {
            'block': ap.StructureClass.parse([
                '....',
                '.##.',
                '.##.',
                '....'
            ]),
            'blinker-h': ap.StructureClass.parse([
                '.....',
                '.###.',
                '.....',
            ]),
            'blinker-v': ap.StructureClass.parse([
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


    def observe(self):
        time = self.environment.get_duration() - 1

        #print()
        print(f"# Observing world at time {time}.")
        
        # single alive cell components
        cells = self.environment.get_cells(time)
        for cell in cells:
            space = {cell}
            self.recognise_component('alive-single', space, time)

        #print("Single alive cell components:", self.components[time]['alive-single'])

        # links between single alive cell components
        alive_single_comps = self.components[time]['alive-single']
        # brute force, but oh well will do for now
        comp_pairs = combinations(alive_single_comps, 2)
        for comp1, comp2 in comp_pairs:
            self.recognise_relation('alive-single-link', comp1, comp2)

        #print("Links between single alive cell components:", self.relations[time]['alive-single-link'])

        # TODO check whether following algo makes more sense via low-level cells-neighbourship instead of relations
        # contingent alive cell components
        alive_link_rels = self.relations[time]['alive-single-link']
        comps_to_check = alive_single_comps.copy()
        groups = []
        while comps_to_check:
            group = []
            groups.append(group)
            # deep search
            comp_agenda = [comps_to_check.pop()]
            while comp_agenda:
                comp = comp_agenda.pop()
                group.append(comp)
                for rel in alive_link_rels:
                    if comp == rel.first:
                        linked_comp = rel.second
                    elif comp == rel.second:
                        linked_comp = rel.first
                    else:
                        continue
                    if linked_comp in comps_to_check:
                        comp_agenda.append(linked_comp)
                        comps_to_check.remove(linked_comp)
        # print("groups", groups)
        # for index, group in enumerate(groups):
        #     print('GROPU', index + 1)
        #     for comp in group:
        #         print(util.set_first(comp.space))
        # group_spaces = [set.union(*[c.space for c in g]) for g in groups]
        # print(group_spaces)
        
        # contingent alive cell components
        for group in groups:
            space = set.union(*[comp.space for comp in group])
            self.recognise_component('alive-contingent', space, time)

        #print("Contingent alive cell components:", self.components[time]['alive-contingent'])

        # FUTURE algo for growing spaces should be somewhere else
        for comp in self.components[time]['alive-contingent']:
            space = frozenset(nb for ce in comp.space for nb in ce.get_neighbours())
            # TODO better merge those components in nested structure?
            self.recognise_component('alive-bounded', space, time)
        
        ##print("Bounded alive cell components:", self.components[time]['alive-bounded'])

        ## FUTURE algo for growing spaces should be somewhere else
        #for comp in self.components[time]['alive-bounded']:
        #    space = frozenset(nb for ce in comp.space for nb in ce.get_neighbours())
        #    # TODO better merge those components in nested structure?
        #    self.recognise_component('alive-bounded-env', space, time)
        #
        #print("Bounded alive cell components with their environment:", self.components[time]['alive-bounded-env'])

        # From here we consider temporal things (that also existed earlier).
        if time == 0:
            return

        # See whether one of the space of one of this time's bounded components
        # is included in the space of one of previous time's bounded components.
        for comp_ab in self.components[time]['alive-bounded']:
            for comp_abe in self.components[time - 1]['alive-bounded']:
                comps_start = frozenset([comp_abe])
                comps_end = frozenset([comp_ab])
                self.recognise_process('bounded-transformation', comps_start, comps_end)

        procs = self.processes['bounded-transformation']
        procs = [p for p in procs if util.set_first(p.end).time == time]
        #print("Bounded transformation processes:", procs)

        
    def reflect(self):

        # FIXME incorporate this somehow into observer memory

        #procs = {proc.start: proc for proc in self.processes['bounded-transformation']}
        #children = {}  # map from node to node list
        #for proc in procs.values():
        #    if proc.start in children:
        #        children[proc.start].append(proc.end)
        #    else:
        #        children[proc.start] = [proc.end]
        #
        #all_children = [node for nodes in children.values() for node in nodes]
        #all_parents = children.keys()
        #parents = {child: parent for (parent, nodes) in children.items() for child in nodes}
        #leaves = [node for node in all_children if node not in all_parents]

        procs = self.processes['bounded-transformation']
        nodes = {}  # tuples of lists of processes, {node: ([incoming, ...], [outgoing, ...]), ...}

        # assumes that procs start and end at different times
        #for proc in procs:
        #    if proc.start in nodes:
        #        nodes[proc.start][1].append(proc)
        #    else:
        #        nodes[proc.start] = ([], [proc])
        #    if proc.end in nodes:
        #        nodes[proc.end][0].append(proc)
        #    else:
        #        nodes[proc.end] = ([proc], [])

        # FIXME clean this up, move to better place
        if True:
            print("Converting processes to graph...")
            graph = util.processes_to_graph(procs)
            if False:
                image_path = '/tmp/debug.png'
                print("Drawing graph...")
                util.write_graph(graph, image_path, separate_components=False, multipartite_layout=True, zoom=1.5)
                import subprocess
                print("Displaying graph...")
                subprocess.run(['kitty', 'icat', image_path])
            if True:
                dest_dir = '/tmp/mai-explore'
                util.write_graph_explorer(graph, dest_dir)

        next_procs = {}  # {proc: [proc, ...]}
        prev_procs = {}  # {proc: [proc, ...]}
        for proc_from in procs:
            for proc_to in procs:
                if proc_from.end == proc_to.start:
                    if proc_from in next_procs:
                        next_procs[proc_from].append(proc_to)
                    else:
                        next_procs[proc_from] = [proc_to]
                    if proc_to in prev_procs:
                        prev_procs[proc_to].append(proc_from)
                    else:
                        prev_procs[proc_to] = [proc_from]

        # for later use
        #unlinked_procs = [p for p in procs if p not in next_procs and p not in prev_procs]
        
        #paths = []
        #for leaf in leaves:
        #    node = leaf
        #    path = []
        #    while node:
        #        path.append(node)
        #        node = parents.get(node, None)
        #    path = list(reversed(path))
        #    paths.append(path)
        
        #print()
        #print("Trajectories of connected processes (identity):")
        #
        #for path_index, path in enumerate(paths):
        #    print(f"Path #{path_index + 1}:")
        #    for index, node in enumerate(path):                
        #        #print(' ' * (2 * index), node)
        #        if node in procs:
        #            print("-", procs[node])

        # deprecated
        # TODO check whether this is inline with theory and if so move elsewhere
        # btw, this is madness
        def is_similar(comp1, comp2):
            def get_bb_min_max(comp):
                loc_iter = iter(cell.location for cell in comp.space)
                loc = next(loc_iter)
                x_min, x_max, y_min, y_max = loc.x, loc.x, loc.y, loc.y
                for loc in loc_iter:
                    if loc.x < x_min: x_min = loc.x
                    if loc.x > x_max: x_max = loc.x
                    if loc.y < y_min: y_min = loc.y
                    if loc.y > y_max: y_max = loc.y
                return x_min, x_max, y_min, y_max
            c1x_min, c1x_max, c1y_min, c1y_max = get_bb_min_max(comp1)
            c2x_min, c2x_max, c2y_min, c2y_max = get_bb_min_max(comp2)
            c1x_delta, c1y_delta = c1x_max - c1x_min, c1y_max - c1y_min
            c2x_delta, c2y_delta = c2x_max - c2x_min, c2y_max - c2y_min
            if c1x_delta != c2x_delta: return False
            if c1y_delta != c2y_delta: return False
            values1 = [None] * ((c1x_delta + 1) * (c1y_delta + 1))
            values2 = [None] * ((c2x_delta + 1) * (c2y_delta + 1))
            for cell in comp1.space:
                dx, dy = cell.location.x - c1x_min, cell.location.y - c1y_min
                index = dy * c1x_delta + dx % c1x_delta
                values1[index] = cell.value
            for cell in comp2.space:
                dx, dy = cell.location.x - c2x_min, cell.location.y - c2y_min
                index = dy * c2x_delta + dx % c2x_delta
                values2[index] = cell.value
            return values1 == values2

        def get_comp_bb_min_max(comp):
            loc_iter = iter(cell.location for cell in comp.space)
            loc = next(loc_iter)
            x_min, x_max, y_min, y_max = loc.x, loc.x, loc.y, loc.y
            for loc in loc_iter:
                if loc.x < x_min: x_min = loc.x
                if loc.x > x_max: x_max = loc.x
                if loc.y < y_min: y_min = loc.y
                if loc.y > y_max: y_max = loc.y
            return x_min, x_max, y_min, y_max

        def get_comp_specs(comp, xy_min_max):
            x_min, x_max, y_min, y_max = xy_min_max
            x_delta, y_delta = x_max - x_min, y_max - y_min
            values = [None] * ((x_delta + 1) * (y_delta + 1))
            for cell in comp.space:
                dx, dy = cell.location.x - x_min, cell.location.y - y_min
                index = dy * x_delta + dx % x_delta
                values[index] = cell.value
            return values

        def get_transition_table(value_domain):
            transition_table = {}
            value_domain_len = len(value_domain)
            for index_from, value_from in enumerate(value_domain):
                for index_to, value_to in enumerate(value_domain):
                    number = index_to * value_domain_len + index_from
                    transition_table[(value_from, value_to)] = number
            return transition_table

        # TODO more elegant way to name and where to put
        # hard-coded value domain for optimisation
        transition_table = get_transition_table([None, True, False])

        def better_hash(obj):
            h = hash(obj)
            h = (h + 2**32) % 2**32
            h = f'{h:08x}'
            return h
        
        # TODO check whether this is inline with theory and if so move elsewhere
        def get_proc_hash(proc):
            # FUTURE current assumptions: start end have single comps, t2-t1=1
            comp1 = util.set_first(proc.start)
            comp2 = util.set_first(proc.end)
            c1_xy_min_max = get_comp_bb_min_max(comp1)
            c2_xy_min_max = get_comp_bb_min_max(comp2)
            x_mins, x_maxs, y_mins, y_maxs = zip(c1_xy_min_max, c2_xy_min_max)
            xy_min_max = min(x_mins), max(x_maxs), min(y_mins), max(y_maxs)
            values1 = get_comp_specs(comp1, xy_min_max)
            values2 = get_comp_specs(comp2, xy_min_max)
            transitions = [transition_table[tuple(pair)] for pair in zip(values1, values2)]
            # hashing could be smarter
            #return str(transitions)
            return better_hash(tuple(transitions))

        # TODO alternative network detection algorithm
        #
        # ALGO v2: find networks of structures
        # - process relations segments is one unit of cycle
        #   - detect repeating NxM process-dependencies
        # - requres deltaT+1 duration of processes to detect cycle
        # - structure-dependency hinges on process-abstraction
        #
        # copy all procs into list of uncyclical procs
        # for each window_size
        #   get all procs from list for window_size+1
        #   get connected sub-graphs for given procs
        #   if start procs equal to end procs
        #     detect maximum stretch of cycle repetitions
        #     remember cycle and stretch
        #     remove procs of stretch from list of uncyclical procs
            
        def get_procs_in_window(start, end, procs):
            for proc in procs:
                if not all(comp.time >= start for comp in proc.start): continue
                if not all(comp.time <= end for comp in proc.end): continue
                yield proc            

        # based on https://stackoverflow.com/a/13837045/8952900
        def get_connected_graph_components(neighbors):
            seen = set()
            def component(node):
                nodes = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    nodes |= neighbors[node] - seen
                    yield node
            for node in neighbors:
                if node not in seen:
                    yield component(node)

        #def get_segment_signature(procs_from, procs_to, next_procs):
        #    # there might be more efficient ways than this
        #    # adjacency-matrix-based solutions?
        #    proc_segment = {proc_hashes[pf]: tuple(proc_hashes[pt] for pt in next_procs[pf] if pt in procs_to) for pf in procs_from}
        #    sig = better_hash(tuple(proc_segment.items()))
        #    return sig

        def get_proc_rel_hash(proc, next_procs):
            proc_hash = proc_hashes[proc]
            next_procs_hashes = '+'.join(proc_hashes[p] for p in next_procs.get(proc, []))
            return f'{proc_hash}-{next_procs_hashes}'

        proc_hashes = {p: get_proc_hash(p) for p in procs}

        print()
        print("Searching for cyclical networks")

        # TODO assume processes that only have start/end comps at the same time
        assert all(len({co.time for co in p.start}) == 1 for p in procs)
        assert all(len({co.time for co in p.end}) == 1 for p in procs)
        # TODO assume all processes stretch only over one unit of time
        assert all(util.set_first(p.end).time - util.set_first(p.start).time == 1 for p in procs)
                    
        last_env_time = self.environment.get_duration()
        # TODO parameterise window size / cycle size / "working memory"
        working_memory_duration = 33
        window_sizes = list(range(2, min(working_memory_duration, last_env_time)))
        unexplained_procs = procs.copy()

        # e.g. a window with size 2 spans over processes at three different times,
        # i.e. one segment (lasting one time unit), capturing 0-* process relationships
        #
        # e.g. window size 3, window start 2 -> times [2, 3, 4, 5] since 5-2=3
        # i.e. 4 sets of components (unused), 3 sets of processes, 2 sets of process relations (segments)
        for window_size in window_sizes:
            print(f"- Cyles of length {window_size - 1}")
            # TODO inner loops should be own function

            # last_env_time is also a valid time
            # because window_size refers to the interval of times, another +1
            for window_start in range(0, last_env_time - (window_size + 1)):
                window_end = window_start + window_size
                procs_window = tuple(get_procs_in_window(window_start, window_end, unexplained_procs))
                # could be optimised
                #next_procs_window = {p: frozenset(next_procs.get(p, [])) for p in procs_window}
                #prev_procs_window = {p: prev_procs[p] for p in procs_window}
                get_links_for = lambda pf: frozenset(pt for pt in next_procs.get(pf, []) + prev_procs.get(pf, []) if pt in procs_window)
                proc_links_window = {pf: get_links_for(pf) for pf in procs_window}
                window_networks = tuple(tuple(s) for s in get_connected_graph_components(proc_links_window))
                #print(f"  - Window {window_start}-{window_end}: Found {len(procs_window)} processes in {len(window_networks)} connected graph components.")
                #util.debug_draw_graph(proc_links_window, f'proc_links_window.wl{window_size}.ws{window_start}')

                #for proc in procs_window:
                #    print(f"    - {proc}")

                # We have networks (connected graph components) within that time window now.
                # When the window_size is N, the networks have an edge depth of (at most) N-1.

                for network_index, window_network in enumerate(window_networks):
                    net_procs = list(window_network)
                    net_segments = {}  # {int: [proc]}
                    for net_proc in net_procs:
                        time = util.set_first(net_proc.start).time
                        if time in net_segments:
                            net_segments[time].add(net_proc)
                        else:
                            net_segments[time] = {net_proc}

                    # A segment is a list of processes for a given interval ("slice"),
                    # based on a given network (i.e. a connected graph component).
                    # Here the interval is referred to by its starting time.
                    
                    # When the window_size is N, there are (at most) N segments.
                    #print(f"    - Segment count is {len(net_segments)}.")

                    # Ignore cases when ends of networks only pertrude a little into the window.
                    # We need at least two segments, otherwise things break.
                    if len(net_segments) == 1:
                        # TODO check again this makes sense
                        #print(f"    - Found network with segment count of 1, skipping. (win:{window_start}-{window_end})")
                        continue
                    
                    # this may break when processes have multiple start/end comps at different times
                    #net_procs_start = [p for p in net_procs if util.set_first(p.start).time == window_start]
                    #net_procs_end = [p for p in net_procs if util.set_first(p.end).time == window_end]
                    net_procs_start = net_segments[min(net_segments)]
                    net_procs_end = net_segments[max(net_segments)]
                    # advance by one time unit, because we are going to compare
                    # the two sets of hashes for process _relations_
                    #net_procs_end = {pt for pf in net_procs_end for pt in next_procs.get(pf, [])}

                    # TODO this is brittle. generally a smarter pattern matching algo would make sense
                    hashes_start = sorted(proc_hashes[p] for p in net_procs_start)
                    hashes_end = sorted(proc_hashes[p] for p in net_procs_end)

                    # TODO re-evaluate this. is this a good way to check for cycles?
                    if hashes_start == hashes_end:
                        # cycle found!
                        hashes_str = '--'.join('/'.join(sorted(proc_hashes[p] for p in net_segments[nsi])) for nsi in sorted(net_segments))
                        print(f"    - Found cycle: {hashes_str}")
                        #print(f"      - Start processes: {net_procs_start}")
                        #print(f"      - End processes:   {net_procs_end}")

                        if True:  # debug info output
                            get_links = lambda pf: {proc_hashes[pt] for pt in next_procs.get(pf, [])}
                            graph_dict = {proc_hashes[pf]: get_links(pf) for pf in window_network}
                            if False:
                                print("CONN")
                                for p in graph_dict:
                                    print(" ", p, graph_dict[p])
                            filename = f'connected_comps.wl{window_size}.ws{window_start}.ni{network_index}'
                            util.debug_draw_graph(graph_dict, filename, verbose=False)
                        
                        for proc in net_procs:
                            try:
                                unexplained_procs.remove(proc)
                            except ValueError:
                                #print("Warning in outer loop, cannot remove:", proc)
                                pass
                        
                        # let's see how far the cycle extends into the future
                        identity_lifetime = window_size
                        procs_cycle = net_procs_start.copy()
                        procs_world = net_procs_end.copy()

                        while procs_cycle and procs_world:
                            # TODO cache hashes for cycle! :C
                            hashes_cycle = sorted(get_proc_rel_hash(p, next_procs) for p in procs_cycle)
                            hashes_world = sorted(get_proc_rel_hash(p, next_procs) for p in procs_world)

                            if hashes_cycle != hashes_world:
                                break

                            for proc in procs_world:
                                try:
                                    unexplained_procs.remove(proc)
                                except ValueError:
                                    #print("Warning in inner loop, cannot remove:", proc)
                                    pass

                            identity_lifetime += 1
                            procs_cycle = {pt for pf in procs_cycle for pt in next_procs.get(pf, [])}
                            procs_world = {pt for pf in procs_world for pt in next_procs.get(pf, [])}

                        print(f"      - Identity extends over {identity_lifetime} time units.")
                    
                    #net_seg_sigs = [get_segment_signature(net_procs[t], net_procs[t + 1], next_procs) for t in range(len(net_segments) - 1)]
                    # len(net_seg_sigs) == window_size

        print()
        print("CONTROLLED STOP OF reflect()")
        return

        # all processes that have no or multiple previous proceses
        traj_starts = [p for p in next_procs if len(next_procs[p]) == 1 and len(prev_procs.get(p, [])) != 1]
        trajectories = []
        
        for traj_start in traj_starts:
            proc = traj_start
            traj = []
            while proc:
                traj.append(proc)
                procs = next_procs.get(proc, [])
                if len(procs) == 1:
                    proc = procs[0]
                else:
                    proc = None
            trajectories.append(traj)

        print("Process trajectories (networks) of identities:")
        print("SKIPPING OUTPUT")
        
        #for index_traj, traj in enumerate(trajectories):
        #    print(f"- Trajectory #{index_traj + 1}")
        #    for index_proc, proc in enumerate(traj):
        #        print(f"  - #{index_proc + 1:02d}: [{proc_hashes[proc]}] {proc}")

        # TODO a process cycle is a network and an own class is warranted
        proc_cycles = []
                
        print()
        print("Cycles in process trajectories (networks):")

        print("WARNING: SKIPPING TRAJECTORIES OF LENGTH 2.")

        # helper to make next part syntactically less verbose
        #musf = lambda p: map(util.set_first, p)
          
        for index_traj, traj in enumerate(trajectories):
            # TODO remove
            if len(traj) == 2: continue
            
            print(f"- Trajectory #{index_traj + 1} (length {len(traj)})")
            proc_cycle = []
            cycle_closed = False
            procs_check = iter(traj)
            procs_ref = iter(traj)
            proc_ref = next(procs_ref)
            for proc_check in procs_check:
                #if is_similar(node_check, node_ref):
                if proc_hashes[proc_check] == proc_hashes[proc_ref]:
                    proc_cycle.append(proc_check)
                    proc_ref = next(procs_ref, None)
                    if not proc_ref:
                        # running out of reference procs to compare with
                        # TODO restart for loop (/nested) starting with other starting reference proc
                        # maybe use offsets/indices to get starting reference proc (cf. triangular matrix)
                        # however, for now this will do. neglecting denerate starting conditions
                        break
                    #if is_similar(node_ref, nodes_cycle[0]):
                    if proc_hashes[proc_ref] == proc_hashes[proc_cycle[0]]:
                        cycle_closed = True
                        proc_cycles.append(proc_cycle)
                        break
            if cycle_closed:
                iter_pair = zip(traj, cycle(proc_cycle))
                #similarities = [is_similar(nc, nr) for (nc, nr) in iter_pair]
                similarities = [proc_hashes[pc] == proc_hashes[pr] for (pc, pr) in iter_pair]
                if all(similarities):
                    print("  - Complete trajectory is cyclical! :)")
                else:
                    print("  - Trajectory is only partially cyclical. :/")
                print(f"  - Cyclical part length is {sum(similarities)}, cycle length is {len(proc_cycle)}.")
            else:
                print("  - No cycle found. :(")

        print()
        print("Process cycles (organisations) found:")

        make_cycle_sig = lambda pc: "-".join(proc_hashes[p] for p in pc)
        cycle_counter = Counter(make_cycle_sig(pc) for pc in proc_cycles)

        for index, (cycle_signature, count) in enumerate(cycle_counter.items()):
            print(f"- Cycle #{index + 1}")
            print(f"  - cycle length: {cycle_signature.count('-') + 1}")
            print(f"  - occurences: {count}")
            print(f"  - signature: {cycle_signature}")

        # TODO cleanup the mess above, divide into separate functions

        # TODO frame process cycles as organisations (traj=net)
        # possibly distinction between as-is process and generalised process is necessary

        # TODO detect cycles not by trajectories, but by moving time windows and graph-component detection
        # TODO then check far cycles extends over time forward
        # - generalise cycle to single process / identity and remove from checking-agenda


    def recognise_all_components(self, kinds=None, times=None):
        if not kinds:
            kinds = self.component_recognisers.keys()
        if not times:
            times = list(range(self.environment.get_duration()))
        for time in times:
            # TODO this only looks at single-cell subspaces
            subspaces = self.environment.get_subspaces((1, 1), time)
            # allow multi-iteration
            subspaces = list(subspaces)
            for kind in kinds:
                #for cell in self.environment.get_cells(time=time):
                #    space = frozenset([cell])
                for space in subspaces:
                    self.recognise_component(kind, space, time)
            # l, t, w, h = self.universe.rects[time]
            # for y in range(t, t + h):
            #     for x in range(l, l + w):
            #         space = frozenset([(x, y)])
            #         for kind in kinds:
            #             self.recognise_component(kind, space, time)

           
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
        decomposed_process = ap.Process(kind, comps_start, comps_end)
        return decomposed_process


    def compose_processes(self, processes):
        comps_start = set(c for p in processes for c in p.start)
        comps_end = set(c for p in processes for c in p.end)
        intersection = comps_start.intersection(comps_end)
        comps_start.difference_update(intersection)
        comps_end.difference_update(intersection)
        kind = f'process-composed'
        composed_process = ap.Process(kind, frozenset(comps_start), frozenset(comps_end))
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
                    clazz = ap.StructureClass.parse(pattern)
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

        structure = ap.Structure(relations)
        self.structures[time][kind].append(structure)

        # quick fix: also store as component of same-named kind
        # FIXME this might rather be done via recognise_component somehow
        space = set()
        space = space.union(*[c.space for c in structure.components()])
        component = ap.Component(kind, frozenset(space), time)
        self.components[time][kind].append(component)
        self.component_structures[component] = structure
        
        return [structure]
        

    # deprecated
    def _is_component_alive_single(self, space, time):
        if len(space) > 1:
            raise ValueError("Multi-cell alive-query not supported.")
        alive = bool(util.set_first(space).value)
        # alive = bool(space.value)
        return alive

    
    def _is_component_alive(self, space, time):
        return all(cell.value for cell in space)


    def _is_component_alive_bounded(self, space, time):
        for component_alive in self.components[time]['alive-contingent']:
            if not space.issuperset(component_alive.space):
                continue
            # comp_alive's space is contained in (arg) space
            boundary = space - component_alive.space
            if any(cell.value for cell in boundary):
                continue
            # no alive cell in boundary
            # TODO should check for contingent space here too
            return True
        return False
    

    # FUTURE merge with above somehow
    def _is_component_alive_bounded_env(self, space, time):
        for component_alive in self.components[time]['alive-bounded']:
            if not space.issuperset(component_alive.space):
                continue
            # TODO should check for contingent space here too
            return True
        return False
    

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
        x1, y1 = util.set_first(comp1.space).location
        x2, y2 = util.set_first(comp2.space).location
        dx, dy = delta
        return x1 - dx == x2 and y1 - dy == y2


    def _recognise_dead_alive_boundary(self, comp1, comp2):
        if len(comp1.space) != 1 or len(comp2.space) != 1:
            raise ValueError("Non atomic space provided!")
        if comp1.time != comp2.time:
            return False
        x1, y1 = util.set_first(comp1.space).location
        x2, y2 = util.set_first(comp2.space).location
        if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
            return False
        return comp1.kind == 'dead' and comp2.kind == 'alive'


    # two neighbouring alive single-cell components
    def _recognise_alive_link(self, comp1, comp2):
        if len(comp1.space) != 1 or len(comp2.space) != 1:
            raise ValueError("Non atomic space provided!")
        if comp1.time != comp2.time:
            return False
        x1, y1 = util.set_first(comp1.space).location
        x2, y2 = util.set_first(comp2.space).location
        if x1 == x2 and y1 == y2:
            return False
        if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
            return False
        return comp1.kind == 'alive-single' and comp2.kind == 'alive-single'

    def _get_neighbourhood_center_component(self, comps):
        pos_x = sum(set_first(c.space)[0] for c in comps) // 9
        pos_y = sum(set_first(c.space)[1] for c in comps) // 9
        assert False
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
        comp_end = util.set_first(comps_end)
        return comp_start.kind == 'dead' and comp_end.kind == 'dead'
            
    def _recognise_process_birth(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = util.set_first(comps_end)
        return comp_start.kind == 'dead' and comp_end.kind == 'alive'

    def _recognise_process_living(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = util.set_first(comps_end)
        return comp_start.kind == 'alive' and comp_end.kind == 'alive'
    
    def _recognise_process_death(self, comps_start, comps_end):
        if len(comps_start) != 9 or len(comps_end) != 1:
            return False
        comp_start = self._get_neighbourhood_center_component(comps_start)
        comp_end = util.set_first(comps_end)
        return comp_start.kind == 'alive' and comp_end.kind == 'dead'

    def _recognise_process_block(self, comps_start, comps_end):
        # we're only supporting single-comp to single-comp processes for now
        if len(comps_start) > 1 or len(comps_end) > 1:
            return False
        comp_start = util.set_first(comps_start)
        comp_end = util.set_first(comps_end)
        return comp_start.kind == 'block' and comp_end.kind == 'block'
    
    def _recognise_process_glider(self, comps_start, comps_end):
        # we're only supporting single-comp to single-comp processes for now
        if len(comps_start) > 1 or len(comps_end) > 1:
            return False
        comp_start = util.set_first(comps_start)
        comp_end = util.set_first(comps_end)
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
    

    def _recognise_process_bounded_transformation(self, comps_start, comps_end):
        # we're only supporting single-comp to single-comp processes for now
        if len(comps_start) > 1 or len(comps_end) > 1:
            return False
        comp_start = util.set_first(comps_start)
        comp_end = util.set_first(comps_end)
        # FIXME the attribute "space" is actually a "spacetime", but shouldn't be.
        # Get the space out of that spacetime, so we can do set operations.
        comp_start_space = {cell.location for cell in comp_start.space}
        comp_end_space = {cell.location for cell in comp_end.space}
        # Grow the starting space so it may contain the end space.
        # TODO fix this for edge cases where comps are only 1 unit width/height
        comp_start_space_ext = {nb.location for ce in comp_start.space for nb in ce.get_neighbours()}
        # TODO check whether this symmetric approach really makes sense! :o
        comp_end_space_ext = {nb.location for ce in comp_end.space for nb in ce.get_neighbours()}
        end_in_start = comp_start_space_ext.issuperset(comp_end_space)
        start_in_end = comp_end_space_ext.issuperset(comp_start_space)
        return end_in_start or start_in_end
    
