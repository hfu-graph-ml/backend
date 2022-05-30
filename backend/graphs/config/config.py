from cmath import isnan
from typing import Tuple, TypedDict
import toml
import os

import utils.checks as checks

# NOTE (Techassi): Maybe move this generic error class into models


class Error:
    def __init__(self, message: str) -> None:
        self.message = message


class ModelOptions(TypedDict):
    emb_size: int
    layer_num_g: int
    num_node_features: int

class TrainingOptions(TypedDict):
    steps_per_epoch: int
    epochs: int

class RewardsOptions(TypedDict):
    step_edge_correct: float
    step_edge_incorrect: float
    terminal_valid_score_min: float
    terminal_valid_score_max: float
    terminal_invalid: float

class DebuggingOptions(TypedDict):
    draw_graph: bool
    print_actions: bool
    print_extracted_features: bool

class Config(TypedDict):
    model: ModelOptions
    training: TrainingOptions
    rewards: RewardsOptions
    debugging: DebuggingOptions


def read(path: str, auto_validate: bool = False) -> Tuple[Config, Error]:
    '''
    Read a TOML file at 'path' and return a new Config class.

    Parameters
    ----------
    path : str
        Path to the TOML config file
    auto_validate : bool
        If this config should be auto validated

    Returns
    -------
    config : Tuple[Config, Error]
        The decoded config or err when error occured
    '''
    if not path:
        return None, Error('Invalid/empty path')

    if not os.path.exists(path):
        return None, Error('File not found')

    try:
        config = toml.load(path, Config)

        if not auto_validate:
            return config, None

        return config, validate(config)
    except toml.TomlDecodeError:
        return None, Error('TOML decode error')


def validate(cfg: Config) -> Error:
    '''
    Validate config values. Returns an error if validation failed.

    Parameters
    ----------
    cfg : Config
        Config to validate

    Returns
    -------
    err : Error
        Non None if validation failed
    '''
    if cfg['model']['emb_size'] < 0:
        return Error('Invalid embedding size.')
        
    if cfg['model']['layer_num_g'] < 0:
        return Error('Invalid layer number.')
        
    if cfg['model']['num_node_features'] < 0:
        return Error('Invalid number of node features.')

    return None
