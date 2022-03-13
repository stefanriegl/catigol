#!/usr/bin/env python3

from collections import defaultdict, Counter
from pprint import pprint

import golly as g

import apgol
import util

# better integration for Golly
import importlib
importlib.reload(util)
importlib.reload(apgol)



GENERATIONS_TO_RUN = 10


# deprecated
# TO DO rename to UniverseHistory
class UniverseRecorder:
    def __init__(self):
        self.states = []
        self.rects = []
    def record(self):
        #rect = g.getrect()
        rect = g.getselrect()
        cells = g.getcells(rect)
        cells = list(zip(cells[0::2], cells[1::2]))
        self.states.append(cells)
        self.rects.append(rect)

        
def golly_get_sel_values(g, rect):
    offset_x, offset_y, width, height = rect
    values = [False] * (width * height)
    
    cells = g.getcells(rect)
    locations = list(zip(cells[0::2], cells[1::2]))

    for abs_x, abs_y in locations:
        index = (abs_x - offset_x) + (abs_y - offset_y) * width
        values[index] = True

    return values


def get_simulation_rect(g):
    rect = g.getselrect()
    if not rect:
        margin = 1
        r_x, r_y, r_w, r_h = g.getrect()
        rect = [r_x - margin, r_y - margin, r_w + 2 * margin, r_h + 2 * margin]
        print("No rectangle selected. Auto-selecting:", rect)
    return rect

    
def main():

    with util.ReportSection("Autopoietic Game of Life", glyph='#'):
        pass

    with util.ReportSection("Setting up and simulating environment."):
        g.reset()
        rect = get_simulation_rect(g)
        env = apgol.GolEnvironment(g, rect)
        env.setup()
        obs = apgol.GolObserver(env)
        obs.observe()

        for gen in range(GENERATIONS_TO_RUN):
            env.simulate_step()
            obs.observe()

        # print(f"Recording {GENERATIONS_TO_RUN}+1 generations... ", end='')
        # env.record_values(golly_get_sel_values(g, rect), 0)
        # # recorder.record()
        # for gen in range(GENERATIONS_TO_RUN):
        #     g.step()
        #     env.record_values(golly_get_sel_values(g, rect), gen + 1)
        #     # recorder.record()
        # print("done.")
        # g.reset()

    with util.ReportSection("Setting up observer."):
        observer = apgol.GliderObserver(env)
        print("Structure classes:", observer.structure_classes.keys())

    with util.ReportSection("Finding structures."):
        structures = {}
        time1 = 0
        time2 = 4
        # times = [time1, time2]
        times = list(range(time1, time2 + 1))
        # print(times)

        for time in times:
            print(f"Processing time {time}.")

            observer.recognise_all_components(times=[time])

            # print(observer.components['alive'])
            # print(observer.components['dead'])
            # input()

            observer.recognise_all_relations(times=[time])

            # print("query")
            # for kind, rels in observer.relations.items():
            #     for rel in rels:
            #         if kind == 'alive-link':
            #             print(rel)
            # input()

            structures = observer.find_structures('block', time)
            structures += observer.find_structures('glider-se-w1', time)
            structures += observer.find_structures('glider-se-w2', time)
            structures += observer.find_structures('glider-se-r1', time)
            structures += observer.find_structures('glider-se-r2', time)

            # print("Found structures:")
            # for structure in structures:
            #     print(f"  {structure.kind}")

            print("Component counts:")
            for key, value in observer.components[time].items():
                print(f"  {key}: {len(value)}")

            # observer.find_structures('glider', time)
            # sg = analysis.get_complex_structures('glider', time)
            # structures[time] = (sa, sd, sg)

            # sa = [x for x in observer.components['alive'] if x.time == time]
            # sd = [x for x in observer.components['dead'] if x.time == time]

            # print('ALIVE: ', sa)
            # print('GLIDER:', sd)

            sa = set()
            sg = set()
            for structure in structures:
                for rel in structure.relations:
                    for comp in (rel.first, rel.second):
                        if comp.kind == 'alive':
                            sa.add(comp)
                        if comp.kind == 'dead':
                            sg.add(comp)

            # rels = [rel for rel in observer.relations['north-of']
            #     if rel.first.time == time and
            #         rel.first.kind == 'alive' and rel.second.kind == 'alive']
            # pprint(rels)

            util.print_unities({
                # 'dot': sd,
                'dotted': sg,
                'full': sa,
            })

            # break
            
            return

    print("### Analysis")
        
    # print("Detecting neighbourhoods...")
    # print(observer.structure_classes['neighbourhood'])
    # for time in times:
    #     observer.find_structures('neighbourhood', time)
    #     print(observer.structures[time])

    # observer.recognise_all_processes(times_start=[time1], times_end=[time2])
    for time in times[:-1]:
        observer.recognise_all_processes(times_start=[time], times_end=[time + 1])
    processes = [p for l in observer.processes.values() for p in l]

    # decompose and replace processes
    if False:
        decomp_procs = []
        for process in processes:
            decomp_proc = observer.decompose_structured_process(process)
            if decomp_proc:
                decomp_procs.append(decomp_proc)
        # specific to processes with only one end-component!
        for process in processes.copy():
            for decomp_proc in decomp_procs:
                if util.set_first(process.end) in decomp_proc.end:
                    processes.remove(process)
        processes += decomp_procs

    # compose and replace processes: per time step
    if True:
        process_groups = defaultdict(list)
        for process in processes:
            # assume process start components all have the same time and are not composed
            comp_start = util.set_first(process.start)
            if comp_start in observer.component_structures:
                continue
            process_groups[comp_start.time].append(process)
            
        # structures = ((t, k, s) for t, d in observer.structures.items() for k, l in d.items() for s in l)
        # for time, kind, structure in structures:
        #     print("testing", time, kind, structure)
        #     # this is a bit sneaky. more roboust solution might be better
        #     # comps = structure.components()
        #     space = set(s for c in structure.components() for s in c.space)
        #     print("space", space)
        #     # struct_procs = [p for p in process_groups[time] if comps.issuperset(p.start)]
        #     struct_procs = [p for p in process_groups[time] if all(space.issuperset(c.space) for c in p.start)]
        #     if len(struct_procs) <= 1:
        #         continue
        #     #...
        
        for time, process_group in process_groups.items():
            groups = util.group_overlapping_processes(process_group)
            for group in groups:
                comp_proc = observer.compose_processes(group)
                for process in group:
                    processes.remove(process)
                processes.append(comp_proc)

        print("Process groups:")
        process_groups_after = defaultdict(list)
        for process in processes:
            # assume process start components all have the same time
            process_groups_after[util.set_first(process.start).time].append(process)
        for key, value in process_groups.items():
            len_after = len(process_groups_after[key])
            print(f"  {key}-{key+1}: {len_after} / {len(value)}")

    # compose and replace processes: over time
    if True:
        groups = util.group_overlapping_processes(processes)
        for group in groups:
            comp_proc = observer.compose_processes(group)
            for process in group:
                processes.remove(process)
            processes.append(comp_proc)

    print("Detected processes:")
    for key, value in observer.processes.items():
        print(f"  {key}: {len(value)}")

    print("Remaining processes:")
    processes_dict = defaultdict(list)
    for process in processes:
        processes_dict[process.kind].append(process)
    for key, value in processes_dict.items():
        print(f"  {key}: {len(value)}")

    print("Composed processes:")
    for index, process in enumerate(processes_dict.get('process-composed', [])):
        counts_start = Counter(c.time for c in process.start)
        counts_end = Counter(c.time for c in process.end)
        keys = set().union(list(counts_start.keys()) + list(counts_end.keys()))
        if True:
            keys = list(range(min(keys), max(keys) + 1))
        else:
            keys = list(sorted(keys))
        counts = {k: (counts_start.get(k, 0), counts_end.get(k, 0)) for k in keys}        
        # print("    start:", Counter(sorted(c.time for c in process.start)))
        # print("    end:", Counter(sorted(c.time for c in process.end)))
        print("   " + "".join(f'      /{v:<3d}' if v else ' ' * 10 for _, v in list(counts.values())[1:]))
        print("   " + "".join(f'     /    '     if v else ' ' * 10 for _, v in list(counts.values())[1:]))
        print("--+-+--".join(f'({k})' for k in keys))
        print("".join(    f'    /     ' if v else ' ' * 10 for v, _ in list(counts.values())[:-1]) + "   ")
        print("".join(f'{v:>3d}/      ' if v else ' ' * 10 for v, _ in list(counts.values())[:-1]) + "   ")
        print()

    # print("Writing graph...")
    # graph = util.processes_to_graph(processes)
    # util.write_graph(graph, '/tmp/ap-process-network.png', multipartite_layout=True)

    # for key, value in observer.processes.items():
    #     if key == 'emptiness':
    #         continue
    #     print(' ', key)
    #     print(value)

    # FIXME theory: this is not optimal
    # unities_in = {ap.Unity(frozenset([p]), time1) for p in ap.set_first(structures[time1][2]).space}
    # unities_out = {ap.Unity(frozenset([p]), time2) for p in ap.set_first(structures[time2][2]).space}
    # movement = observer.process('glider-travel-4', unities_in, unities_out)

    # print('MOVEMENT:', movement)

    #    from pprint import pprint
    #    pprint(observer.unities['alive'])

    
    print("done")


    
main()
