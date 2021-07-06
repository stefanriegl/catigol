#!/usr/bin/env python3

from pprint import pprint

import golly as g

import ap
import util

# better integration for Golly
import importlib
importlib.reload(ap)


GENERATIONS_TO_RUN = 10


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


if not len(g.getselrect()):
    print("Error: No rectangle selected. I can't work like that.")
    
else:    
    print()
    print("####")
    g.reset()
    recorder = UniverseRecorder()

    print(f"Recording {GENERATIONS_TO_RUN} generations... ", end='')
    recorder.record()
    for _ in range(GENERATIONS_TO_RUN):
        g.step()
        recorder.record()
    print("done.")


    print("Analyzing...")
    observer = ap.Observer(recorder)
    analysis = ap.AutopoieticAnalysis(observer)

    # print("Structure classes of the observer:")
    # print(observer.structure_classes)

    structures = {}
    time1 = 0
    time2 = 4
    times = [time1, time2]

    for time in times:
        analysis.recognise_components('alive', time)
        analysis.recognise_components('dead', time)

        # print(observer.components['alive'])
        # print(observer.components['dead'])
        # input()
        
        for kind in observer.relation_recognisers:
            analysis.recognise_relations(kind, time)

        # print("query")
        # for kind, rels in observer.relations.items():
        #     for rel in rels:
        #         if kind == 'alive-link':
        #             print(rel)
        # input()
            
        structures1 = observer.find_structures('block', time)
        structures2 = observer.find_structures('glider-se-w1', time)
        structures = structures1 + structures2

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

        print(structures)
                    
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

    # FIXME theory: this is not optimal
    # unities_in = {ap.Unity(frozenset([p]), time1) for p in ap.set_first(structures[time1][2]).space}
    # unities_out = {ap.Unity(frozenset([p]), time2) for p in ap.set_first(structures[time2][2]).space}
    # movement = observer.process('glider-travel-4', unities_in, unities_out)

    # print('MOVEMENT:', movement)

    #    from pprint import pprint
    #    pprint(observer.unities['alive'])

    
