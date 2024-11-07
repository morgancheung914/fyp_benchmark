# Introduction

This repository is intended for the benchmarking of FYP project.

## Setup 

1. Build openssl from source
```
cd 
mkdir ssl 
mkdir sslsrc 
cd sslsrc

wget www.openssl.org/source/openssl-1.1.1.tar.gz
tar xf openssl-1.1.1.tar.gz
cd openssl-1.1.1
./config --prefix=$HOME/openssl --openssldir=$HOME/openssl 
make 
make install
```

2. Add these to .bashrc by  vim ~/.bashrc
```
export PATH="$HOME/openssl/bin:$PATH"
export LDFLAGS="-L$HOME/openssl/lib"
export CPPFLAGS="-I$HOME/openssl/include"
export PKG_CONFIG_PATH="$HOME/openssl/lib/pkgconfig"
export LD_LIBRARY_PATH="$HOME/openssl/lib:$LD_LIBRARY_PATH"
```

3. Reload Shell 
```
source ~/.bashrc
```


4. Install pyenv 
``` 
curl https://pyenv.run | bash

```

5. Add to  ~/.bashrc
``` 
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

6. Reload Shell 
```
source ~/.bashrc
```

7. Installation of python 3.11
```
pyenv install 3.11
pyenv global 3.11.10
```

8. Create Virtual environment

```
python3 -m venv benchenv
source ~/benchenv/bin/activate
```


9. Install depedencies
```
# install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

10. Running
`python src/llama3.py`



## Install new dependencies

If you need to install new dependencies, the preferred way is to use `python3 -m pip install`

## slurm configs 
config: `export SLURM_CONF=/opt1/slurm/gpu-slurm.conf`


`srun -p gpu_2h -c 4 --gres=gpu:2 -C "rtx3090" --pty /bin/bash`
or 
`srun -p gpu_2h -c 4 --gres=gpu:2 -C "rtx2080" --pty /bin/bash`
or when the queue is long 
`srun -p gpu_2h -c 4 --gres=gpu:8 --pty /bin/bash`

## Using tmux to spawn multiple shells within the same job

First run `tmux`.

To create new pane run `Ctrl-b %`.

To navigate between panes use `Ctrl-b o` or `Ctrl-b <arrows>`

To detach from session use `Ctrl-b d`

To kill session use `Ctrl-b :` to open command mode and write `kill-session`


## How to use the shorten script 

write the following into ~/.bashrc

`export GROQ_API_KEY=your_key_here`

Then use 
`python process_response.py -c config.yaml`, specifying the response paths to be shortened in response.chosen_datsets and the few_shot parameter as True or False.


The script will create shortened/ and include the shortened dataset.

## Cycle through multiple Groq API_keys

A shell script is created for the ease of replacing API_keys once they run out of quota, simply run
`./autoshort_3.sh`
In the script, replace these few things:
1. groq.txt -> a txt file path that contains the API keys
2. log_file= -> a log file path 
3. config_4.yaml -> a yaml file you wish to use on process_response.py
