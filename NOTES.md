# TODOs
- node embedding 0 when nodes are isolated at beginning
- valuenet / critic design
- pretrain critic

# Future Improvements
- add more metrics to tensorboard
  - valid action ratio
  - valid table ratio
  - mood score
  - some graph plots/figures
- extend config
- use git commit hash in model id to associate trained models with its codebase
- some form of changelog / training documentation
- use profiler for training speed improvements
- -> start training with all nodes and features
- maybe use hyperopt?
- improve code structure and add explaining comments for next project group

# Commands
- `tensorboard --logdir ./tensorboard_logs`
- `source .venv/Scripts/activate`
- `snakeviz profiling_stats/stats.profile`

# Comparisons
- PPO_39 (first node picked is masked out in second prob dist) vs PPO_41 (can pick same node but then gets negative reward)
- PPO_41 (max_steps=100) vs PPO_42 (max_steps=50)
- PPO_41 (max_steps=100 and rewards 1.0/10.0 and layer_num=4) vs PPO_44 (max_steps=100 and rewards 0.25/25.0 and layer_num=2)
- PPO_41 (max_steps=100 and rewards 1.0/10.0 and layer_num=4) vs PPO_45 (max_steps=100 and rewards 1/-0.25/10/-10 and layer_num=4)
- PPO_45 (rewards 1/-1/10/-10) vs PPO_ (rewards 1/-1/20/-0 and fixed and scaled score function to [.3, .7])
- PPO_47 fixed mood_score and rewards 1/-1/20/-10
- PPO_52 allow more bad steps, handle structural validity at end of epoch, add vec env 8xDummyVecEnv
- 2022-06-07_08-51-24 reverse PPO_52 changes to PPO_47, keep vec env
- test_temp/PPO2 (rewards 0.5/-0.5/20/-20 and 4xDummyVecEnv) vs 2022-06-07_08-51-24 (rewards 1/-1/20/-10)