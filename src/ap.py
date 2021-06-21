
from collections import defaultdict
from itertools import permutations, product


def set_first(s):
    for e in s:
        return e
    return None


def print_unities(unities_dict):
    grid = defaultdict(lambda: defaultdict(list))

    for char, unities in unities_dict.items():
        for unity in unities:
            for x, y in unity.space:
                grid[y][x] = char

    min_y = min(grid.keys())
    max_y = max(grid.keys())

    min_xs, max_xs = zip(*((min(r), max(r)) for r in grid.values()))
    min_x = min(min_xs)
    max_x = max(max_xs)
            
    min_pos = str((min_x, min_y))
    max_pos = str((max_x, max_y))
    hr = '-' * (max_x - min_x + 1)
    
    print(f"{min_pos}")
    print(f"+{hr}+")

    for y in range(min_y, max_y + 1):
        print('|', end='')
        for x in range(min_x, max_x + 1):
            cell = grid[y][x]
            if not cell:
                cell = ' '
            print(cell, end='')
        print('|')
    
    print(f"+{hr}+")
    print(f"{max_pos:>{len(hr)+2}}")


class AutopoieticAnalysis:

    def __init__(self, observer):
        self.observer = observer

        
    def get_simple_structures(self, kind, time=None):
        
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
            for y in range(t, t + h):
                for x in range(l, l + w):
                    unity = Unity(set([(x, y)]), time_)
                    if self.observer.prop(kind, unity):
                        unities.add(unity)

        return unities


    # FIXME only glider for now
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
                    if -5 <= x <= -3 and 2 <= y <= 4: pass
                    else: continue
                    print(f"now considering: ({x}, {y})")
                    space = set(product(range(x, x + 5), range(y, y + 5)))
                    # TODO add rotated biting away corners
                    space.remove((x + 0, y + 0))
                    space.remove((x + 0, y + 1))
                    space.remove((x + 4, y + 0))
                    unity = Unity(space, time_)
                    if self.observer.prop('glider', unity):
                        unities.add(unity)

        return unities

    
class Unity:
    
    def __init__(self, space, time):
        self.space = set(space)
        self.time = time
        
    def __iter__(self):
        return iter((self.space, self.time))
    
#    def __contains__(self, other):
#        if self.time != other.time:
#            return False
#        return all(p in self.space for p in other.space)

    def __repr__(self):
        return f"<U:{self.space}@{self.time}>"
    

class Observer:

    def __init__(self, universe):
        self.universe = universe
        # "memory"
        self.unities = defaultdict(list)

        
    def _prop_alive(self, unity):
        space, time = unity
        
        if len(space) > 1:
            raise ValueError("Multi-cell alive-query not supported.")

        alive = (set_first(space) in self.universe.states[time])
        return alive


    def _prop_glider(self, unity):
        space, time = unity

        if len(space) != 22:
            return False

        alive = [u for u in self.unities['alive'] if u.space.issubset(space)]

        if len(alive) != 5:
            return False
        
        dead = [u for u in self.unities['dead'] if u.space.issubset(space)]

        print_unities({
            '●': alive,
            '◌': dead
        })

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
            
            print("yas relations spacing ###############")
            
            # now let's check that the right cells are not specified
            if any(self.relation('south-west-west-of', u1, ud) for ud in dead): continue
            print("yas d1")
            if any(self.relation('south-east-east-of', u1, ud) for ud in dead): continue
            print("yas d2")
            if any(self.relation('south-south-east-of', u5, ud) for ud in dead): continue
            # success! we have a glider in .:, orientation

            print("yas yas yas")
            
            return True
        
        # no permutation of suitably arranged alive cells could be found
        return False

        
    def prop(self, kind, unity):
        
        if kind == 'alive':
            state = self._prop_alive(unity)
            # FIXME side effect. should be put elsewhere
            if state:
                self.unities['alive'].append(unity)
            return state

        # for completeness
        if kind == 'dead':
            state = not self._prop_alive(unity)
            # FIXME side effect. should be put elsewhere
            if state:
                self.unities['dead'].append(unity)
            return state

        if kind == 'glider':
            state = self._prop_glider(unity)
            # FIXME side effect. should be put elsewhere
            if state:
                self.unities['dead'].append(unity)
            return state
        
        raise ValueError("Invalid kind specified: " + kind)

        
    def relation(self, kind, unity1, unity2):

        geography = {
            'west-of': (-1, 0),
            'east-of': (+1, 0),
            'north-of': (0, +1),
            'south-of': (0, -1),
            'north-west-of': (-1, +1),
            'north-east-of': (+1, +1),
            'south-west-of': (-1, -1),
            'south-east-of': (+1, -1),
            'north-north-west-of': (-1, +2),
            'north-north-east-of': (+1, +2),
            'north-west-west-of': (-2, +1),
            'north-east-east-of': (+2, +1),
            'south-west-west-of': (-2, -1),
            'south-east-east-of': (+2, -1),
            'south-south-west-of': (-1, -2),
            'south-south-east-of': (+1, -2),
        }

        if kind in geography.keys():
            space1, time1 = unity1
            space2, time2 = unity2
            if len(space1) != 1 or len(space2) != 1:
                raise ValueError("Non atomic space provided!")
            if time1 != time2:
                return False
            x1, y1 = set_first(space1)
            x2, y2 = set_first(space2)
            dx, dy = geography[kind]
            # inverted y
            return x1 - dx == x2 and y1 + dy == y2
            
        raise ValueError("Invalid kind specified: " + kind)
