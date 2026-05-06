"""
loads config files and substitutes keywords
"""
# external imports
import yaml
from collections import namedtuple
from random import randint

# local imports
from infill_workflow.utils import (
    validate,
    files,
    strings,
    display,
)

def load(path_str: str, debug=False):
    """
    loads a yaml config file and substitues the keywords with pre-set values

    :param path_str: the path of the config file
    :type param: str
    """

    validate.path_exists(path_str)
    keywords = {
        'home' : files.home(),
        'project_root' : files.project_root(),
        'timestamp' : strings.now()
    }

    base_cfg = {
        'api_keys': {},
        'datasets': {},
        'resp_coll': {},
        'eval': {},
        'unit_test': {},
        'model_params': {},
        'infill': {},
    }
    categories = base_cfg.keys()

    with open(path_str, 'r') as f:
        cfg = f.read()

    cfg = strings.replace_slots(
        cfg,
        keywords
    )

    # convert to named tuple
    cfg = {**base_cfg, **yaml.safe_load(cfg)}
    cfg = namedtuple('Config', categories)(**cfg)

    # set default values
    cfg = set_defaults(cfg, keywords, debug)

    if debug:
        pass

    return cfg, keywords

def set_defaults(cfg, keywords, debug=False):
    return cfg

def check_required(cfg):
    # check if trainer is assigned
    try:
        cfg.general['trainer']
    except Exception as e:
        display.error('config parameter cfg.general[\'trainer\'] is not set')
