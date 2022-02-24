#!/bin/bash

cd "$(dirname "$0")"
source "venv/bin/activate"
GTK_THEME=Adwaita:light golly "test-world.mc"

