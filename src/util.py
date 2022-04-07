
from collections import defaultdict
from itertools import product
from shutil import copy as shutil_copy
import os.path
import sys
from json import dumps as json_dumps
from subprocess import run as subprocess_run
from threading import Thread

#from ap import StructureClass, ComponentRelationConstraint

import networkx as nx
from matplotlib import pyplot as plot
# import subprocess
import pandas as pd


class ReportSection:
    def __init__(self, header=None, footer=None, glyph='*', width=78):
        self.width = width
        self.glyph = glyph
        self.header = header
        self.footer = footer
    def _print_box(self, text):
        print(self.glyph * self.width)
        print('{{}} {{:{}s}} {{}}'.format(self.width - 4).format(self.glyph, text, self.glyph))
        print(self.glyph * self.width)
    def __enter__(self):
        if self.header:
            print()
            self._print_box(self.header)
    def __exit__(self, type_, value, traceback):
        if self.footer:
            self._print_box(self.footer)


def set_first(s):
    for e in s:
        return e
    #return None
    raise ValueError("Empty set!")


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
            for cell in unity.space:
                x, y = cell.location
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


def get_comps_and_procs_graph(processes):
    graph = nx.DiGraph()
    component_names = {}

    for index, process in enumerate(processes):

        time_start = max(c.time for c in process.start)
        time_end = min(c.time for c in process.end) if process.end else time_start + 1
        time = (time_start + time_end) / 2

        process_name = f'{process.kind}-p{index + 1}'
        graph.add_node(process_name, category='process', time=time, entity=process, index=index)
        
        for components, start in ((process.start, True), (process.end, False)):
            for component in components:
                
                if component in component_names:
                    component_name = component_names[component]
                else:
                    component_index = len(component_names) + 1
                    component_name = f'{component.kind}-c{component_index}'
                    component_names[component] = component_name
                    graph.add_node(component_name, category='component', time=component.time, entity=component, index=component_index)

                if start:
                    node_start, node_end = component_name, process_name
                else:
                    node_start, node_end = process_name, component_name

                graph.add_edge(node_start, node_end)

        # print(process)
        # break

    # print(graph.nodes())
    # print(graph.edges())
    # print(component_names)

    return graph


def get_procs_graph(graph_dict):
    # graph_dict is a {node: [node,...]} adjecency dict-of-lists
    graph = nx.DiGraph()
    component_names = {}

    procs = frozenset([n for ns in graph_dict.values() for n in ns] + list(graph_dict.keys()))
    node_ids = {}
    links = [(n1, n2) for (n1, ns) in graph_dict.items() for n2 in ns]

    for index, proc in enumerate(sorted(procs, key=lambda p: set_first(p.start).time)):
        node_id = f'{proc.kind}-p{index + 1}'
        node_ids[proc] = node_id
        
        time_start = max(c.time for c in proc.start)
        time_end = min(c.time for c in proc.end) if proc.end else time_start + 1
        time = (time_start + time_end) / 2

        graph.add_node(node_id, category='process', time=time, entity=proc, index=index)

    for pf, pt in links:
        graph.add_edge(node_ids[pf], node_ids[pt])

    return graph


def write_graph(graph, output_file_path, separate_components=True, do_labels=True, zoom=5, multipartite_layout=True):
    ##  for page in pages:
    ##    graph.add_node(page.id)
    ##    for id in re_findall(export.Page.regex_link, page.text):
    ##      graph.add_edge(page.id, id)
    
    # pos = nx.nx_pydot.graphviz_layout(graph, 'neato', None)
    # pos = nx.nx_pydot.pydot_layout(graph, 'sfdp', None)
    # pos = nx.spring_layout(graph, k=1)
    # pos = nx.multipartite_layout(graph, subset_key='time', align='vertical')
    # pos = nx.kamada_kawai_layout(graph)
    # pos = nx.bipartite_layout(graph, nodes_component)

    if separate_components:
        subgraphs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
    else:
        subgraphs = [graph]

    padding_y = None
    pos = {}

    # separate components (subgraphs) vertically
    for subgraph in sorted(subgraphs, key=len, reverse=True):
        if multipartite_layout:
            subpos = nx.multipartite_layout(subgraph, subset_key='time', align='vertical')
        else:
            # subpos = nx.spring_layout(subgraph, k=1)
            subpos = nx.nx_pydot.graphviz_layout(subgraph, 'dot', None)
        if pos:
            # determine component padding based on first component (biggest)
            values_y = [y for x, y in pos.values()]
            values_y_min = min(values_y)
            if not padding_y:
                padding_y = 0.2 * (max(values_y) - values_y_min)
                if not padding_y:
                    padding_y = 0.1  # dump quick fix
            offset_y = values_y_min - padding_y - max(y for (x, y) in subpos.values())
            # print("O", values_y_min, padding_y, offset_y)
            # for coord in subpos.values():
            #     coord[1] += offset_y
            subpos = {k: (x, y + offset_y) for k, (x, y) in subpos.items()}
        pos.update(subpos)

    if multipartite_layout:
        # fix horizontal node positioning such that nodes for all components align
        for node, data in graph.nodes(data=True):
            # pos[node][0] = data["time"]
            pos[node] = (data["time"], pos[node][1])

    # # TODO debug output, move this elsewhere
    # print("Component positions:")
    # for node, data in sorted(graph.nodes(data=True), key=lambda t: t[1]['index']):
    #     if data['category'] != 'component':
    #         continue
    #     space = data['entity'].space
    #     position = set_first(space)
    #     print(f'  {node}: {position}')

    zoom *= 4 * 3
    plot.figure(figsize=(2 * zoom, 2 * zoom))

    # nodes
    nodes_process = [n for n, d in graph.nodes(data=True) if d['category'] == 'process']
    nodes_component = [n for n, d in graph.nodes(data=True) if d['category'] == 'component']
    node_groups = [nodes_process, nodes_component]
    colours = ['#ffff88', '#ff88ff']
    shapes = 'so'
    for nodes, colour, shape in zip(node_groups, colours, shapes):
        colours = [colour] * len(nodes)
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_shape=shape, node_color=colours)

    # edges
    colours = []
    for node1, node2 in graph.edges():
        node_proc = node1 if graph.nodes[node1]['category'] == 'process' else node2
        kind = graph.nodes[node_proc]['entity'].kind
        colour = '#cccccc' if kind == 'emptiness' else '#000000'
        colours.append(colour)
    nx.draw_networkx_edges(graph, pos, width=1, edge_color=colours)

    # labels
    if do_labels:
        labels = {}
        for node, data in graph.nodes(data=True):
            if data['category'] == 'component':
                space = data['entity'].space
                if len(space) == 1:
                    position = set_first(data['entity'].space)
                else:
                    poss_x, poss_y = zip(*(cell.location for cell in space))
                    pos_x = sum(poss_x) / len(poss_x)
                    pos_y = sum(poss_y) / len(poss_y)
                    position = f"({pos_x:.2f}, {pos_y:.2f})"
                label = f"{node}\n{position}"
            else:
                label = node
            labels[node] = label
        # bbox_options = {"ec": "k", "fc": "white", "alpha": 0.5}
        # nx.draw_networkx_labels(graph, pos, labels=labels, font_size=4, bbox=bbox_options)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=4, font_color='#000088')
        
    plot.axis('off')
    # plot.show()

    #  nx.draw(graph, pos=None, node_color='#A0CBE2', edge_color='none', width=1, edge_cmap=plot.cm.Blues, with_labels=False)
    #  plot.savefig("graph.png", dpi=500, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1) 
    # nx.draw_spring(graph, node_color='#A0CBE2', linewidths=0, edge_color='#cccccc', width=1, with_labels=True)
    plot.savefig(output_file_path, dpi=200, bbox_inches='tight')
    # plot.show()
    plot.close('all')


def debug_draw_graph(graph_dict, filename_base, verbose=True):
    #edge_args = 'color="grey", penwidth=3.0'
    edge_args = ''

    all_nodes = frozenset([n for ns in graph_dict.values() for n in ns] + list(graph_dict.keys()))
    nids = {n: i for (i, n) in enumerate(all_nodes)}
    graph_edges = [(n1, n2) for (n1, ns) in graph_dict.items() for n2 in ns]

    def get_node_props(obj):
        if type(obj) == str:
            label = obj
        else:
            label = repr(obj)
        label = label.replace(' bounded-transformation', ' ba')
        label = label.replace(' alive-contingent', '')
        label = label.replace(' {', '\n{')
        props = []
        props.append(f'label="{label}"')
        #props.append(f'pos="{pos_x},{pos_y}"')
        return ', '.join(props)

    nodes_str = '\n'.join(f'{nids[n]} [ {get_node_props(n)} ];' for n in all_nodes)
    edges_str = '\n'.join(f'{nids[n1]} -> {nids[n2]} [ {edge_args} ];' for (n1, n2) in graph_edges)
    graph_str = f'digraph {{ rankdir=TB;\n{nodes_str}\n{edges_str}\n}}'
    
    out_path = f'/tmp/mai-debug/{filename_base}.png'
    args = ['/usr/bin/dot', '-Tpng', f'-o{out_path}']

    if verbose:
        print(f"Writing debug graph '{filename_base}'... ", end='')

    #print(graph_str)
    proc = subprocess_run(args, text=True, input=graph_str, capture_output=True)

    if verbose:
        if proc.returncode:
            print(f"error {proc.returncode}.")
            for line in proc.stderr.split('\n'):
                print(line)
        else:
            print("done.")

# def write_graph(edges, output_file_path):
# #  orientation="landscape"
# #  bgcolor="transparent"
# #  edge [penwidth="1.2"]
# #  node [penwidth="1.2"]
# #  node [fontname="sans"]
#     graph_elems = ["""
#         digraph G {
#             size="6,10"
#             dpi="100"
#             rankdir=LR
#             bgcolor="black"
#             edge [color="green"]
#             node [color="green"]
#             node [fontcolor="green"]
#             node [fontname="Arial"]
#     """]
#     graph_elems.extend(map(lambda n: f'{n[0]} [label="{n[1]}"];\n', nodes))
#     graph_elems.extend(map(lambda e: f'{e[0]} -> {e[1]};\n', edges))
#     graph_elems.append('}')
#     graph_def = ''.join(graph_elems)
#     proc = subprocess.Popen(['/usr/bin/neato', '-o%s' % output_file_path, '-Tpng'], stdin=subprocess.PIPE)
#     proc.communicate(input=graph_def)
#     proc.wait()


def group_overlapping_processes(processes):
    proc_by_comp = defaultdict(list)
    for process in processes:
        for comps in (process.start, process.end):
            for comp in comps:
                proc_by_comp[comp].append(process)
    processes_left = processes.copy()
    processes_agenda = []
    groups = []
    while processes_left:
        group = []
        groups.append(group)
        processes_agenda = [processes_left.pop()]
        while processes_agenda:
            proc = processes_agenda.pop()
            group.append(proc)
            for comps in (proc.start, proc.end):
                for comp in comps:
                    for proc_comp in proc_by_comp[comp]:
                        if proc_comp in processes_left:
                            processes_left.remove(proc_comp)
                            processes_agenda.append(proc_comp)
    return groups


def write_graph_explorer(graph, dest_dir):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    index_path = f'{dest_dir}/index.html'
    #if not os.path.exists(index_path):
    script_dir = os.path.dirname(__file__)
    shutil_copy(f'{script_dir}/explorer.html', index_path)

    #node_groups = {
    #    'component': 1,
    #    'process': 2,
    #}
    #node_group_default = 0
    comps_data = [d for (_, d) in graph.nodes(data=True) if d['category'] == 'component']
    time_max = max(d['time'] for d in comps_data)

    # TODO inefficient, optimise
    locs = [ce.location for d in comps_data for ce in d['entity'].space]
    min_x = min(loc.x for loc in locs)
    max_x = max(loc.x for loc in locs)
    min_y = min(loc.y for loc in locs)
    max_y = max(loc.y for loc in locs)

    # TODO quick and dirty hack, move to better place
    def format_id(text):
        #text = text.replace('bounded-transformation', 'bt')
        #text = text.replace('alive-bounded', 'ab')
        tokens = text.split('-')
        abbr = ''.join(t[0] for t in tokens[:-1])
        text = f"{abbr}-{tokens[-1]}"
        return text
    
    def gen_node(node, data):
        #group = node_groups.get(data['category'], node_group_default)
        assert data['category'] in ('process', 'component')
        if data['category'] == 'process':
            proc = data['entity']
            comps = frozenset.union(proc.start, proc.end)
            space = frozenset.union(*(comp.space for comp in comps))
        else:
            space = data['entity'].space
        cells_count = len(space)
        center_x = sum(ce.location.x for ce in space) / cells_count
        center_y = sum(ce.location.y for ce in space) / cells_count
        return {
            'id': format_id(node),
            #'group': group
            'category': data['category'],
            'time': data['time'] / time_max,
            'center': {
                'x': (center_x - min_x) / (max_x - min_x),
                'y': (center_y - min_y) / (max_y - min_y),
            }
        }

    def gen_edge(node1, node2):
        return {
            'source': format_id(node1),
            'target': format_id(node2),
            'value': 1,
        }
        
    data = {}
    data['nodes'] = [gen_node(n, d) for n, d in graph.nodes(data=True)]
    data['links'] = [gen_edge(n1, n2) for n1, n2 in graph.edges()]

    for time in (0.0, 1.0):
        nodes_time = [n for n in data['nodes'] if n['time'] == time]
        nodes_count = len(nodes_time)
        if nodes_count > 0:
            nodes_sorted = sorted(nodes_time, key=lambda n: n['center']['y'])
            for index, node in enumerate(nodes_sorted):
                node['ratio_y'] = index / (nodes_count - 1)
        
    with open(f'{dest_dir}/data.json', 'w') as f:
        f.write(json_dumps(data))

        
class ErrorStreamRedirect:
    def __init__(self, target):
        self._target = target
        self._original = None
    def __enter__(self):
        self._original = sys.stderr
        sys.stderr = self._target
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.stderr = self._original

        
def send_graph_to_cytoscape(graph, network_title, background=False):
    import py4cytoscape as p4c
    from datetime import datetime as dt

    try:
        # Golly prints messages that went to stderr in a popup window.
        # Redirect stderr to stdout in this case, i.e. fail silently,
        # if Cytoscape is just not running.
        with ErrorStreamRedirect(sys.stdout):
            p4c.cytoscape_ping()
    except:
        print("Cannot ping Cytoscape, not exporting.")
        return

    collection_name = "CATIGOL"
    date_str = dt.now().strftime('%F %H:%M:%S')
    network_name = f"{network_title} - {date_str}"

    def conv(obj):
        return obj if type(obj) in [str, int, bool, float] else repr(obj)

    def render_name(text):
        tokens = text.split('-')
        abbr = ''.join(t[0] for t in tokens[:-1])
        text = f"{abbr}-{tokens[-1]}"
        return text

    if False:
        nodes = graph.nodes(data=True)
        nodes = [(n, {k: conv(v) for (k, v) in d.items()}) for (n, d) in nodes]

        edges = graph.edges(data=True)
        edges = [(n1, n2, {k: conv(v) for (k, v) in d.items()}) for (n1, n2, d) in edges]

        graph2 = nx.DiGraph()
        graph2.add_nodes_from(nodes)
        graph2.add_edges_from(edges)

        try:
            print(f"Exporting '{network_title}' to Cytoscape... ", end='', flush=True)
            p4c.create_network_from_networkx(graph2, title=network_name, collection=collection_name)
            print("done.")
        except ex:
            print("error.")
            print(ex)

    # Let's take a detour via Pandas to make sure data types are exported correctly.

    nodes = graph.nodes(data=True)
    nodes = [(render_name(n), d) for (n, d) in nodes]
    nodes = [(n, {k: conv(v) for (k, v) in d.items()}) for (n, d) in nodes]
    nodes = [{**attrs, 'name': name} for name, attrs in nodes]
    nodes_df = pd.DataFrame.from_records(nodes)

    edges = graph.edges(data=True)
    edges = [(render_name(n1), render_name(n2), d) for (n1, n2, d) in edges]
    edges = [(n1, n2, {k: conv(v) for (k, v) in d.items()}) for (n1, n2, d) in edges]
    edges = [{**attrs, 'source': n1, 'target': n2} for n1, n2, attrs in edges]
    edges_df = pd.DataFrame.from_records(edges)

    nodes_df['name'] = nodes_df['name'].astype(str)
    edges_df['source'] = edges_df['source'].astype(str)
    edges_df['target'] = edges_df['target'].astype(str)
    
    if 'interaction' in edges_df.columns:
        edges_df['interaction'] = edges_df['interaction'].astype(str)

    if len(nodes_df.index) == 0: nodes_df = None
    if len(edges_df.index) == 0: edges_df = None

    if background:
        # export in background

        def do_export():
            print(f"Starting export of '{network_title}' to Cytoscape in background.")
            try:
                p4c.create_network_from_data_frames(
                    nodes=nodes_df, edges=edges_df, title=network_name, collection=collection_name,
                    node_id_list='name')
                print("Export to Cytoscape in background finished successfully.")
            except:
                print("Error: Export to Cytoscape in background failed.")
        
        Thread(target=do_export).start()
    else:
        try:
            print(f"Exporting '{network_title}' to Cytoscape... ", end='', flush=True)
            p4c.create_network_from_data_frames(
                nodes=nodes_df, edges=edges_df, title=network_name, collection=collection_name,
                node_id_list='name')
            print("done.")
        except Exception as ex:
            print("error.")
            print(ex)
