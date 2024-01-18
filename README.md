# diffusion-nbs
Getting started with diffusion

This is a fork of https://github.com/fastai/diffusion-nbs that provides a Pipfile with dependencies, as well as instructions for running locally in Ubuntu, either on bare metal or in WSL2. An NVIDIA GPU is required in either case (CUDA works in WSL2). I've tested these instructions on 22.04 running in WLS2 on Windows 11, and in 23.10 running on bare metal.

## Set up your environment

Clone this repo:
```
git clone https://github.com/jason-weddington/diffusion-nbs.git
```

CD into the newly cloned directory:
```
cd diffusion-nbs
```

Install pip3 and pipenv (if not already installed)
```
sudo apt -y install python3-pip
pip3 install --user pipenv
```

Create a new Python virtual environment in repo directory.
```
pipenv shell
```

Install dependencies:
```
pipenv install
```

Run Jupyter Lab:
```
jupyter lab
```

## Huggingface API Token

Create a Huggingface account then generate an access token from our settings page. You'll need this to download pre-trained models from Huggingface. The access.

https://huggingface.co/settings/tokens

## Run the notebooks

Browse to http://localhost:8888/lab/tree/stable_diffusion.ipynb and start running the cells in the notebook.

## Shut down

Close Jupyter Lab by pressing CTRL + C in the terminal window where it is running.

