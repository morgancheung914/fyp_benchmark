# Introduction

This repository is intended for the benchmarking of FYP project.

## Setup 

Install pyenv 
``` 
curl https://pyenv.run | bash

```

Add to  ~/.bashrc
``` 
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload Shell 
```
source ~/.bashrc
```

Installation of python 3.11
```
pyenv install 3.11
pyenv global 3.11.10
pyenv which python #record this path
```
Create Virtual environment
```
virtualenv --python=paste_path_here benchenv .

source benchenv/bin/activate
```

Install depedencies
```
# install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```



## Install new dependencies

If you need to install new dependencies, the preferred way is to use `python3 -m pip install`

## slurm configs 
config: `export SLURM_CONF=/opt1/slurm/gpu-slurm.conf`


`srun -p gpu_24h --gres=gpu:1 -C "rtx3090" --pty /bin/bash`