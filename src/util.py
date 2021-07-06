
from collections import defaultdict


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
    

# could be optimised, if it were important

def pattern_rotate90(pattern):
    return pattern_transpose(pattern_mirror_y(pattern))

def pattern_rotate180(pattern):
    return pattern_mirror_x(pattern_mirror_y(pattern))

def pattern_rotate270(pattern):
    return pattern_mirror_y(pattern_transpose(pattern))

def pattern_transpose(pattern):
    return [''.join(l) for l in zip(*pattern)]

def pattern_mirror_x(pattern):
    return [''.join(reversed(l)) for l in pattern]

def pattern_mirror_y(pattern):
    return list(reversed(pattern))
