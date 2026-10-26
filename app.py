#!/usr/bin/env python3
"""One-command launcher for the UAV Strategy Engine web dashboard.

Usage::

    python app.py                 # http://127.0.0.1:5000 (opens a browser)
    python app.py --port 8000     # custom port
    python app.py --no-browser    # headless / server use
"""

from __future__ import annotations

from uav_strategy.webapp import main

if __name__ == "__main__":
    raise SystemExit(main())
