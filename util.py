"""
Helpful functions.
"""


def flatten(l):
    if isinstance(l, tuple):
        return sum(map(flatten, l), [])
    else:
        return [l]