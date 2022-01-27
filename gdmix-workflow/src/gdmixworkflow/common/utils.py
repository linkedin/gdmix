from collections import namedtuple

import yaml


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


def rm_backslash(params):
    """ A '-' at the beginning of a line is a special charter in YAML,
    used backslash to escape, need to remove the added backslash for local run.
    """
    return {k.strip('\\'): v for k, v in params.items()}


def yaml_config_file_to_obj(config_file):
    """ load gdmix config from yaml file to object. """
    def _yaml_object_hook(d):
        return namedtuple('GDMIX_CONFIG', d.keys())(*d.values())

    with open(config_file) as f:
        config_obj = _yaml_object_hook(yaml.safe_load(f))
    return config_obj
