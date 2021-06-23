#!/usr/bin/env python3

import golly as g

import ap

# better integration for Golly
import importlib
importlib.reload(ap)


GENERATIONS_TO_RUN = 10


class UniverseRecorder:
    def __init__(self):
        self.states = []
        self.rects = []
    def record(self):
        rect = g.getrect()
        cells = g.getcells(rect)
        cells = list(zip(cells[0::2], cells[1::2]))
        self.states.append(cells)
        self.rects.append(rect)

    
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

structures = {}
time1 = 0
time2 = 4
times = [time1, time2]

for time in times:
    sa = analysis.get_simple_structures('alive', time)
    sd = analysis.get_simple_structures('dead', time)
    sg = analysis.get_complex_structures('glider', time)
    structures[time] = (sa, sd, sg)

#    print('ALIVE: ', sa)
#    print('GLIDER:', sg)
    
    ap.print_unities({
        'dot': sd,
        'dotted': sg,
        'full': sa,
    })

# FIXME theory: this is not optimal
unities_in = {ap.Unity(frozenset([p]), time1) for p in ap.set_first(structures[time1][2]).space}
unities_out = {ap.Unity(frozenset([p]), time2) for p in ap.set_first(structures[time2][2]).space}
movement = observer.process('glider-travel-4', unities_in, unities_out)

print('MOVEMENT:', movement)
    
#    from pprint import pprint
#    pprint(observer.unities['alive'])
