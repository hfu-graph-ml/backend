from cmath import isnan
import os
import json
import copy
import numpy as np
import networkx as nx
from networkx.readwrite import edgelist
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat, Figure

from config import config
from utils import draw_graph


class ConfigLoggerCallback(BaseCallback):
    '''
    Callback for printing the configuration file used to train the model to the TensorBoard log.

    Args:
        config: Configuration as dict.
    '''

    def __init__(self, config=config):
        super(ConfigLoggerCallback, self).__init__(verbose=0)
        self.config = config

    def _on_training_start(self) -> bool:
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in self.logger.output_formats if isinstance(
                formatter, TensorBoardOutputFormat))
        self.tb_formatter.writer.add_text("info/config", str(json.dumps(self.config, indent=2)
                                                                                                                 ).replace('  ', '&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n', '  \n'), self.num_timesteps)
        self.tb_formatter.writer.flush()
        return True

    def _on_step(self) -> bool:
        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    '''
    Callback for saving a model (the check is done every ``check_freq`` steps) based on the training reward.

    Args:
        log_dir: Path to the folder where the model will be saved. It must contains the file created by the ``Monitor`` wrapper.
        check_freq: Frequency of steps in which to check if a better model can be saved. Defaults to number of steps per epoch.
        verbose: Verbosity level.
    '''

    def __init__(self, log_dir: str, check_freq: int = config["training"]["steps_per_epoch"], verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class LogValidActions(BaseCallback):
    '''
    Callback for logging the ratio of valid actions the model takes to the TensorBoard log.

    Args:
        check_freq: Frequency of steps in which to log a new value. Defaults to number of steps per epoch.
    '''

    def __init__(self, check_freq: int = config["training"]["steps_per_epoch"]):
        super(LogValidActions, self).__init__()
        self.check_freq = check_freq
        self.valid_actions = []

    def _on_step(self) -> bool:
        self.valid_actions.append([x["action_valid"] for x in self.locals["infos"]])

        if self.n_calls % self.check_freq == 0:
            _, counts = np.unique(self.valid_actions, return_counts=True)
            valid_action_ratio = counts[1] / (counts[0] + counts[1])
            self.logger.record('rollout/valid_action_ratio', valid_action_ratio)
            self.valid_actions = []

        return True


class LogValidGraphs(BaseCallback):
    '''
    Callback for logging the ratio of valid graphs the model generates to the TensorBoard log.

    Args:
        check_freq: Frequency of steps in which to log a new value. Defaults to number of steps per epoch.
    '''

    def __init__(self, check_freq: int = config["training"]["steps_per_epoch"]):
        super(LogValidGraphs, self).__init__()
        self.check_freq = check_freq
        self.valid_graphs = []

    def _on_step(self) -> bool:
        self.valid_graphs.append([x["graph_valid"] for x in self.locals["infos"]])

        if self.n_calls % self.check_freq == 0:
            u, c = np.unique(self.valid_graphs, return_counts=True)
            counts = dict(zip(u, c))
            n_valid = counts[1] if 1 in counts.keys() else 0
            n_invalid = counts[0] if 0 in counts.keys() else 0
            valid_graph_ratio = n_valid / (n_invalid + n_valid)
            self.logger.record('rollout/valid_graph_ratio', valid_graph_ratio)
            self.valid_graphs = []

        return True


class LogMoodScores(BaseCallback):
    '''
    Callback for logging the mean mood score of valid graphs the model generates to the TensorBoard log. NaN if the model didn't generate any valid graphs.

    Args:
        check_freq: Frequency of steps in which to log a new value. Defaults to number of steps per epoch.
    '''
    
    def __init__(self, check_freq: int = config["training"]["steps_per_epoch"]):
        super(LogMoodScores, self).__init__()
        self.check_freq = check_freq
        self.mood_scores = []

    def _on_step(self) -> bool:
        mood_scores = [x["mood_score"] for x in self.locals["infos"]]
        self.mood_scores.append(mood_scores)

        if self.n_calls % self.check_freq == 0:
            mean_mood_score = np.nanmean(self.mood_scores)
            self.logger.record('rollout/mean_mood_score', mean_mood_score)
            self.mood_scores = []

        return True


class LogBestGraph(BaseCallback):
    '''
    Callback for logging and keeping track of the highest and lowest mood scores of valid graphs the model generates to the TensorBoard log.
    The best valid graph will also be visualized as figure.

    Args:
        check_freq: Frequency of steps in which to log a new value. Defaults to number of steps per epoch.
    '''
    
    def __init__(self, check_freq: int = config["training"]["steps_per_epoch"]):
        super(LogBestGraph, self).__init__()
        self.check_freq = check_freq
        self.highest_mood_score = None
        self.lowest_mood_score = None
        self.got_new_highest_mood_score = False
        self.highest_mood_score_graph = None
        self.tb_formatter = None

    def _on_training_start(self) -> bool:
        self.tb_formatter = next(formatter for formatter in self.logger.output_formats if isinstance(
                formatter, TensorBoardOutputFormat))
        return True

    def _on_step(self) -> bool:
        mood_scores = [x["mood_score"] for x in self.locals["infos"]]

        highest_mood_score_index = np.argmax(np.nan_to_num(mood_scores))
        highest_mood_score = mood_scores[highest_mood_score_index]
        if not isnan(highest_mood_score) and (self.highest_mood_score == None or highest_mood_score > self.highest_mood_score):
            self.highest_mood_score = highest_mood_score
            self.highest_mood_score_graph = copy.deepcopy(self.locals["infos"][highest_mood_score_index]["graph"])
            self.got_new_highest_mood_score = True
        
        
        lowest_mood_score_index = np.argmin(np.nan_to_num(mood_scores, nan=1000))
        lowest_mood_score = mood_scores[lowest_mood_score_index]
        if not isnan(lowest_mood_score) and (self.lowest_mood_score == None or lowest_mood_score < self.lowest_mood_score):
            self.lowest_mood_score = lowest_mood_score


        if self.n_calls % self.check_freq == 0:
            self.logger.record('rollout/highest_mood_score', self.highest_mood_score)
            self.logger.record('rollout/lowest_mood_score', self.lowest_mood_score)

            if self.got_new_highest_mood_score:
                fig = draw_graph(self.highest_mood_score_graph, show_graph=False)
                self.logger.record("graphs/highest_mood_score_graph", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
                graph_edge_list_string = ""
                for edge in edgelist.generate_edgelist(self.highest_mood_score_graph, delimiter=", ", data=False):
                    graph_edge_list_string = graph_edge_list_string + f"({edge}),&nbsp;"
                graph_edge_list_string = graph_edge_list_string[:-7]
                self.tb_formatter.writer.add_text("graphs/highest_mood_score_graph", graph_edge_list_string, self.num_timesteps)
                self.tb_formatter.writer.flush()
                self.got_new_highest_mood_score = False

        return True
