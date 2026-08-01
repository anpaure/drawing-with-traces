#!/usr/bin/env python3
"""Draw one PNG with real model-training power and automatic refinement."""

from __future__ import annotations

import sys

from drawing_with_traces.cli import main


if __name__ == "__main__":
    main(["draw-png", *sys.argv[1:]])
