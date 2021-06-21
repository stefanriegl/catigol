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

#print('rect[0] ==', recorder.rects[0])
sa = analysis.get_simple_structures('alive', 0)
print('ALIVE', sa)
sd = analysis.get_simple_structures('dead', 0)
#print('DEAD', sd)
ap.print_unities({
    '#': sa,
    '.': sd,
})

g = analysis.get_complex_structures('glider', 0)
print(g)

