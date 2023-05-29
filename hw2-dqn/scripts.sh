python -m papermill code.ipynb code_minigrid.ipynb -f config_train_minigrid.yaml
python -m papermill code.ipynb code_pong.ipynb -f config_train_pong.yaml -y "{devices:[0]}"
