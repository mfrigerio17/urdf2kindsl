#!/usr/bin/env python3

import sys, pathlib

# Trick the python path to use the modules in the local source
# tree, to try the tool without installation.
# Note however that the runtime dependencies will still be needed.

pathHere = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathHere / 'src'))

from urdf2kindsl import cmdline

if __name__ == '__main__':
    cmdline.main()
