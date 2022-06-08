from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import json


class ConfigLoggerCallback(BaseCallback):
  def __init__(self, config, verbose=0):
    super(ConfigLoggerCallback, self).__init__(verbose)
    self.config = config

  def _on_training_start(self) -> bool:
    # Save reference to tensorboard formatter object
    # note: the failure case (not formatter found) is not handled here, should be done with try/except.
    self.tb_formatter = next(formatter for formatter in self.logger.output_formats if isinstance(
        formatter, TensorBoardOutputFormat))
    self.tb_formatter.writer.add_text("info/config", str(json.dumps(self.config, indent=2)).replace('  ', '&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n', '  \n'), self.num_timesteps)
    self.tb_formatter.writer.flush()
    return True

  def _on_step(self) -> bool:
    return True
