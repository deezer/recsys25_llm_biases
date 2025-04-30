# Here lives your code

from pathlib import Path

# Path to the data dir, for loading files
DATA_DIR = (Path(__file__).parent / "data").resolve()


def some_function(value):
    return str(value)
