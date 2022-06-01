# TODOs
- reward design (vs rules?)
  - compare
  - when is graph generation done?
  - how to enforce 1st node is not 2nd node?
  - how to enforce edge doesn't already exist?
  - how to enforce correct graph structure? (max node degrees, ...)
    - immediate rewards for max degree of node? or validity check at end for terminal reward?
- node embedding 0 when nodes are isolated at beginning
- valuenet / critic design
- pretrain critic

# Notes
- fewer nodes
- old graph und falsche actions verhindern
- max steps

# Commands
- `tensorboard --logdir ./tensorboard_logs/model_trained_test`

# Comparisons
- PPO_39 (first node picked is masked out in second prob dist) vs PPO_41 (can pick same node but then gets negative reward)
- PPO_41 (max_steps=100) vs PPO_42 (max_steps=50)
- PPO_41 (max_steps=100 and rewards 1.0/10.0 and layer_num=4) vs PPO_44 (max_steps=100 and rewards 0.25/25.0 and layer_num=2)
- PPO_41 (max_steps=100 and rewards 1.0/10.0 and layer_num=4) vs PPO_45 (max_steps=100 and rewards 1/-0.25/10/-10 and layer_num=4)
- PPO_45 (rewards 1/-1/10/-10) vs PPO_ (rewards 1/-1/20/-0 and fixed and scaled score function to [.3, .7])