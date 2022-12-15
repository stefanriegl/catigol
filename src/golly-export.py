#!/usr/bin/env python3

from subprocess import run as sp_run
from PIL import Image

import golly as g


EXPORT_STDOUT = False
EXPORT_CLIPBOARD = False
EXPORT_IMAGE = True

EXPORT_IMAGE_PATH = "/tmp/gol-export.{:04d}.png"
EXPORT_GENERATIONS = list(range(30))


def export_rect(rect):
  offset_x, offset_y, width, height = rect
  values = [[False] * width for _ in range(height)]
  cells = g.getcells(rect)
  locations = list(zip(cells[0::2], cells[1::2]))

  for abs_x, abs_y in locations:
    index_x = abs_x - offset_x
    index_y = abs_y - offset_y
    values[index_y][index_x] = True

  if EXPORT_STDOUT or EXPORT_CLIPBOARD:
    rect_str = "\\goldiagram{\n"

    for row in values:
      row_str = ",".join("1" if cell else "0" for cell in row)
      rect_str += f"  {{{row_str}}},\n"
    rect_str += "}\n"

    if EXPORT_STDOUT:
      print(rect_str)

    if EXPORT_CLIPBOARD:
      print("Exporting to clipboard.")
      cmdline = "xclip -selection clipboard -i".split(" ")
      sp_run(cmdline, input=rect_str, text=True)

  if EXPORT_IMAGE:
    path = EXPORT_IMAGE_PATH.format(int(g.getgen()))
    print("Exporting to file: " + path)
    data = bytes(255 * cell for row in values for cell in row)
    im = Image.frombytes("L", (width, height), data)
    im = im.convert("RGB")
    im.save(path)


rect = g.getselrect()

if rect:
  if not EXPORT_GENERATIONS:
    export_rect(rect)
  else:
    gen_min = min(EXPORT_GENERATIONS)
    gen_max = max(EXPORT_GENERATIONS)
    gen = int(g.getgen())
    if gen > gen_min:
      g.reset()
      gen = 0
    while gen < gen_min:
      g.step()
      gen += 1
    while gen <= gen_max:
      if gen in EXPORT_GENERATIONS:
        export_rect(rect)
      g.step()
      gen += 1
    g.reset()
else:
  print("No source area selected.")

