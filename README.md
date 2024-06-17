# MissileCombat
The missile combat gym simulation environment based on Jsbsim
## Structure
Todo

## Install 

```shell
# create python env
conda create -n jsbsim python=3.8
# install dependency
pip install torch pymap3d jsbsim geographiclib gym==0.20.0 wandb icecream setproctitle. 

- Download Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely), and `pip install shaply` from local file.

- Initialize submodules(*JSBSim-Team/jsbsim*): `git submodule init; git submodule update`
```
## Envs
provide all task configs in  `envs/JSBSim/configs`, each config corresponds to a task.
### SingleControl
SingleControl env includes single agent heading task, whose goal is to train agent fly according to the given direction, altitude and velocity. The trained agent can be used to design baselines or become the low level policy of the following combat tasks. We can designed two baselines, as shown in the video:

### SingleCombat
Todo
### MultipleCombat
Todo

## Quick Start
### Training

```bash
# on linux
cd scripts/sh
bash train_*.sh
# on windows
cd scripts/bat
.\train_*.bat
```
We have provide scripts for linux in `scripts/sh` and scripts for windows in `scripts/bat`.

- `train_heading.sh` or `train_heading.bat` is for SingleControl environment heading task.
- `train_vsbaseline.sh` or `train_vsbaseline.bat` is for SingleCombat vs-baseline tasks.
- `train_selfplay.sh` or `train_selfplay.bat` is for SingleCombat self-play tasks. 
- `train_selfplay_shoot.sh` or `train_selfplay_shoot.bat` is for SingleCombat self-play shoot missile tasks.
- `train_share_selfplay.sh` or `train_share_selfplay.bat` is for MultipleCombat self-play tasks.

It can be adapted to other tasks by modifying a few parameter settings. 

- `--env-name` includes options ['SingleControl', 'SingleCombat', 'MultipleCombat'].
- `--scenario` corresponds to yaml file in `envs/JBSim/configs` one by one.
- `--algorithm` includes options [ppo, mappo], ppo for SingleControl and SingleCombat, mappo for MultipleCombat

The description of parameter setting refers to `config.py`.
Note that we set parameters `--use-selfplay --selfplay-algorithm --n-choose-opponents --use-eval --n-eval-rollout-threads --eval-interval --eval-episodes` in selfplay-setting training. `--use-prior` is only set true for shoot missile tasks.
We use wandb to track the training process. If you set `--use-wandb`, please replace the `--wandb-name` with your name. 

### Evaluate and Render
```bash
cd renders
python render*.py
```

This will generate a `*.acmi` file. We can use [**TacView**](https://www.tacview.net/), a universal flight analysis tool, to open the file and watch the render videos.
