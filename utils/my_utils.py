"""
This file contains utility functions that might be used.
"""
import munch
import torch
import inspect


def fix_args_if_needed(args):
    if not isinstance(args, dict):
        return args
    for a in args:
        if isinstance(args[a], str) \
                and ('e-' in args[a]):
            args[a] = float(args[a])
        if isinstance(args[a], str) \
                and ('e+' in args[a]):
            args[a] = int(args[a])
        args[a] = fix_args_if_needed(args[a])
    return args

