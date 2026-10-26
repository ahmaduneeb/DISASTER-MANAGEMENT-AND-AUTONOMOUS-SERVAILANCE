"""Allow ``python -m uav_strategy ...``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
