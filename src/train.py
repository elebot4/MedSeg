"""Compatibility training entrypoint.

Keeps the public command `python src/train.py <config.py> [--key=value]`
while delegating to the existing implementation in base_train.py.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("base_train.py")), run_name="__main__")
