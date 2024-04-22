import yaml
from typing import Dict


def parse_yaml_cfg(cfg_path: str) -> Dict:
    """Parse yaml config file to dictionary

    Args:
        cfg_path (str): file path

    Returns:
        Dict: config in dictionary format
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
