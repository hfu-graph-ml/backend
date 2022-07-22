# TODOs
- debug & verify node embeddings, ...
- calc grid layout positions for draw graph
- scale training data (randomly generate more nodes to draw from)
- some form of changelog / training documentation
- extend config?
- use hyperopt?
- improve code structure and add explaining comments for next project group

# Notes
- failing in completing some pre-connected graphs
  - model 2022-06-28_16-12-03:
    - (9, 2) -> macht probleme wenn 9 kein anderes edge hat
  - model 2022-06-09_14-21-02:
    - (3, 4)
    - (3, 4), (1, 3)
  - model 2022-06-16_15-03-11:
    - [(0, 3), (0, 4), (4, 2)] -> 0-2/100 (5.5)
    - [(0, 3), (0, 4), (4, 2), (4, 5)] -> 55-71/100 (5.5)
    - [(0, 3), (0, 4), (4, 2), (3, 5)] -> 0-3/100 (5.5)
  - model 2022-06-16_20-37-13:
    - [(0, 3), (0, 4), (4, 2)] -> 37-44/100 (8.0)
    - [(0, 3), (0, 4), (4, 2), (4, 5)] -> 26-33/100 (5.5)
    - [(0, 3), (0, 4), (4, 2), (3, 5)] -> 55-63/100 (6.5)

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
- 2022-06-16_11-28-14 (normalized terminal score rewards) vs 2022-06-09_14-21-02
- 2022-06-16_15-03-11 (layer_num=1 and steps_per_epoch=4000) vs 2022-06-16_11-28-14
- 2022-06-16_20-37-13 (train with preconnect_nodes_probability .5) vs 2022-06-16_15-03-11
- 2022-06-17_10-25-47 (20 nodes) vs 2022-06-16_20-37-13 (6 nodes)
- 2022-06-17_12-18-04 (rewards 0.1/-0.1/20/-20 and max_steps=500) vs 2022-06-17_10-25-47 (rewards 0.5/-0.5/20/-20 and max_steps=100)
- 2022-06-17_12-51-10 / 2022-06-18_09-33-22 (rewards 0.25/-0.25/20/-20 and max_steps=250) vs 2022-06-17_12-18-04 (rewards 0.1/-0.1/20/-20 and max_steps=500)
- 2022-06-17_15-56-53 (rewards 0.5/-0.5/20/-20 and max_steps=200 and preconnect_nodes_probability=0.25) vs 2022-06-17_12-51-10 (rewards 0.25/-0.25/20/-20 and max_steps=250 and preconnect_nodes_probability=0.5)
- 2022-06-17_19-53-32 (rewards 0.5/-0.5/50/-20 and max_steps=150 and preconnect_nodes_probability=0.05) vs 2022-06-17_15-56-53 (rewards 0.5/-0.5/20/-20 and max_steps=200 and preconnect_nodes_probability=0.25)
- 2022-06-18_09-33-22 (2022-06-17_12-51-10 model for longer)
- train 2022-06-17_10-25-47 model for longer (did i do that?)
- 2022-06-27_17-39-05 (10 nodes and rewards 0.25/-0.25/20(12.5-22.5)/-20 and max_steps=250 and preconnect_nodes_probability=0.5)
- 2022-06-27_20-09-24 (rewards 0.5/-0.5/20(12.5-22.5)/-20 and max_steps=200) vs 2022-06-27_17-39-05
- 2022-06-28_09-50-50 (emb_size=4) vs 2022-06-27_17-39-05 (emb_size=8)
- 2022-06-28_16-12-03 (emb_size=16 and rewards 0.5/-0.5/20(9-26.5)/-20 and max_steps=150 and preconnect_nodes_probability=0.25)

# Best models
- for 6 nodes: 2022-06-16_20-37-13
- for 10 nodes:
  - 2022-06-28_16-12-03
  - 2022-06-28_09-50-50
    - probleme:
      - 4-5
      - 4-5_1-0
- for 20 nodes: 2022-06-17_10-25-47?