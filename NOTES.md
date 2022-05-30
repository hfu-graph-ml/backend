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