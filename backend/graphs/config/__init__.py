from typing import Tuple, TypedDict
import toml
import os
import click


class Error:
    def __init__(self, message: str) -> None:
        self.message = message


class DataOptions(TypedDict):
    num_nodes: int
    num_node_features: int

class ModelOptions(TypedDict):
    emb_size: int
    layer_num_g: int

class TrainingOptions(TypedDict):
    max_steps: int
    steps_per_epoch: int
    num_training_envs: int
    batch_size: int
    epochs: int

class RewardsOptions(TypedDict):
    step_edge_correct: float
    step_edge_incorrect: float
    terminal_valid_score_min: float
    terminal_valid_score_max: float
    terminal_invalid: float

class DebuggingOptions(TypedDict):
    draw_graph: bool
    draw_correct_graphs: bool
    print_actions: bool
    print_extracted_features: bool

class Config(TypedDict):
    data: DataOptions
    model: ModelOptions
    training: TrainingOptions
    rewards: RewardsOptions
    debugging: DebuggingOptions


def read(path: str) -> Tuple[Config, Error]:
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
        return config, None
    except toml.TomlDecodeError:
        return None, Error('TOML decode error')

# Load config
config_path = os.path.dirname(os.path.realpath(__file__)) + '/example.toml'
config, err = read(config_path)
if err != None:
  click.echo(f'Failed to load config \'{config_path}\': {err}')