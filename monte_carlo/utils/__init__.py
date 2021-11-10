from ..resources import values
from . import data

def round_floats(o):
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def trim_list(untrimmed_list):
    if len(untrimmed_list) <= values.list_length:
        return untrimmed_list
    else:
        return untrimmed_list[-values.list_length:]
