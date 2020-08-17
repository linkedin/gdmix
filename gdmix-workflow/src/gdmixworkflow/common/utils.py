from collections import namedtuple
from gdmixworkflow.common.constants import *
from itertools import product
import json
from subprocess import Popen,PIPE,STDOUT


def gen_random_string(length=6):
    """ Generate fixed-length random string. """
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def abbr(name):
    """ Return abbreviation of a given name.
    Example:
        fixed-effect -> f10t
        per-member -> p8r
    """
    return "{}{}{}".format(name[0], len(name) - 2,
                           name[-1]) if len(name) > 2 else name


def prefix_dash_dash(params):
    """ Add -- for keys in gdmix tfjob params. """
    if isinstance(params, dict):
        newParams = {}
        for k, v in params.items():
            newParams["--{}".format(k)] = v
        return newParams
    else:
        raise ValueError("job params can only be dict")


def join_params(params):
    """ Join param to string as key value pair. If the key begins
    with '#', the key is ignored.
    """
    if isinstance(params, dict):
        return (' ').join(["{} {}".format(k, v) if not k.startswith(
            '#') else str(v) for (k, v) in params.items()])
    else:
        raise ValueError("job params can only be dict")


def rm_backslash(params):
    """ A '-' at the begining of a line is a special charter in YAML,
    used backslash to escape, need to remove the added backslach for local run.
    """
    newParams = {}
    for k, v in params.items():
        newParams[k.strip('\\')] = v
    return newParams


def json_config_file_to_obj(config_file):
    """ load gdmix config from json file to object. """
    def _json_object_hook(d):
        return namedtuple('GDMIX_CONFIG', d.keys())(*d.values())

    with open(config_file) as f:
        config_obj = json.load(f, object_hook=_json_object_hook)
    return config_obj


def flatten_config_obj(d, config_obj):
        """ flatten a config obj to a dict without nested dict.
        For example: {a: {b: c}} --> {b: c}
        """
        for k, v in config_obj._asdict().items():
            if type(v) not in [str, bool, int, float, list]:
                flatten_config_obj(d, v)
            else:
                d[k] = v
        return d