"""
A wrapper to extract_1d.extract_1d for a science object reduction.
"""

# import all here so setup.py can be on-boarded
# in the top scope. TODO: find a better solution for this.
from extract_1d import *

if __name__ == "__main__":
    extract_1d_spec(standard_star=False)
