import json, yaml
from collections import namedtuple


def gen_random_string(length=6):
    """ Generate fixed-length random string. """
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def abbr(name):
    """ Return abbreviation of a given name.
    Example:
        fixed-effect -> f10t
        per-member -> p8r
    """
    return name if len(name) <= 2 else f"{name[0]}{len(name) - 2}{name[-1]}"


def prefix_dash_dash(params):
    """ Add -- for keys in gdmix tfjob params. """
    if isinstance(params, dict):
        return {f"--{k}": v for k, v in params.items()}
    else:
        raise ValueError("job params can only be dict")


def join_params(params):
    """ Join param to string as key value pair. If the key begins
    with '#', the key is ignored.
    """
    if isinstance(params, dict):
        return ' '.join(f"{k} {v}" if not k.startswith('#') else str(v) for k, v in params.items())
    else:
        raise ValueError("job params can only be dict")


def rm_backslash(params):
    """ A '-' at the beginning of a line is a special charter in YAML,
    used backslash to escape, need to remove the added backslash for local run.
    """
    return {k.strip('\\'): v for k, v in params.items()}


def json_config_file_to_obj(config_file):
    """ load gdmix config from json file to object. """
    def _json_object_hook(d):
        # return d
        # d = {k: _json_object_hook(v) if type(v) is dict else v for k, v in d.items()}
        return namedtuple('GDMIX_CONFIG', d.keys())(*d.values())

    with open(config_file) as f:
        config_obj = _json_object_hook(yaml.safe_load(f))
    return config_obj
