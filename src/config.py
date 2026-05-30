"""
Poor Man's Config - borrowed from nanoGPT
Example usage:
$ python train.py config/2d_axi.py --batch_size=4 --learning_rate=0.001
this will first run config/override_file.py, then override batch_size to 4 and learning rate to 0.001

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        # assume it's the name of a config file
        assert not arg.startswith("--")
        config_file = arg
        # print(f"Loading config from {config_file}:")
        exec(open(config_file).read())
    else:
        # assume it's a --key=value override
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]  # remove --

        # attempt to eval it (e.g. if bool, number, or etc)
        try:
            # handle special cases
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                val = literal_eval(val)
        except (ValueError, SyntaxError):
            # if that goes wrong, just use the string
            pass

        print(f"Overriding: {key} = {val}")
        # Update global namespace
        globals()[key] = val
