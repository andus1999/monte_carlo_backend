from ..resources import values


def round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(x) for x in obj]
    return obj


def trim_list(untrimmed_list):
    if len(untrimmed_list) <= values.list_length:
        return untrimmed_list
    else:
        return untrimmed_list[-values.list_length:]
