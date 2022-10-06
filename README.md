# Direct Advantage Estimation
Official repository for "Direct Advantage Estimation"

## Requirements

We recommend using Python 3.8 with venv. Please make sure 
pip is up to date by running:

```bash
pip install -U pip
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Training

To reproduce the results, run the following command:
```bash
python train.py --algo {algo} --hparam_file {hyperparameter_file} --envs {env} --threads {threads}
```

`--algo`: `PPO` (GAE) or `CustomPPO` (DAE)

`--hparam_file`: See `./params/` for the hyperparameters used in the paper, the files are named by `{algo}_{network}.yml`

`--envs`: Environment to train. For example, `Pong`, `Breakout`, etc. For MinAtar environments, please add the suffix `-MinAtar-v0`. (e.g., `Breakout-MinAtar-v0`)

### Optional arguments

`--threads`: Number of parallel threads for asynchronous environment steps

`--logging`: Save logs in `./logs/{env}/`

`--save_model`: Save the trained model to `./logs/{env}/`

## Viewing logs

To view the tensorboard logs, run
```bash
python -m tensorboard --logdir ./logs/
```
and open the displayed URL in a browser.

## How to cite

Please use the following BibTex entry.

```
@article{pan2021direct,
  title={Direct Advantage Estimation},
  author={Pan, Hsiao-Ru and G{\"u}rtler, Nico and Neitz, Alexander and Sch{\"o}lkopf, Bernhard},
  journal={arXiv preprint arXiv:2109.06093},
  year={2021}
}
```





