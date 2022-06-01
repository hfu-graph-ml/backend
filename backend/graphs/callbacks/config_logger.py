from stable_baselines3.common.callbacks import BaseCallback

class ConfigLoggerCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super(ConfigLoggerCallback, self).__init__(verbose)
        self.config = config

    def _on_training_start(self):
        self.logger.record("info/config", self.config)
        return True