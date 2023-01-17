###### tags: `Tools` `Python` `conda` `DataScience` 
## Install
<details>
    <summary><em>Miniconda</em></summary>

# Miniconda
1. 到[官方網站](https://docs.conda.io/en/latest/miniconda.html)下載對應版本
    - e.g.
        ```sh=
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        chmod +x ./Miniconda3-latest-Linux-x86_64.sh
        ./Miniconda3-latest-Linux-x86_64.sh
        ```
2. 以root執行安裝，並指定位置為 **/usr/local/miniconda3/**
3. 在shell (以vim開啟~/.bashrc或~/.zshrc) 中加入以下環境變數:
```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

4. (optional) 執行以下指令來取消預設啟動base環境

```bash
conda config --set auto_activate_base false
```

</details>
<details>
    <summary><em>Miniforge</em></summary>

# Miniforge
- https://developer.apple.com/metal/tensorflow-plugin/
- Can work with Mac m1
```shell
__conda_setup="$('/usr/local/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
```
</details>


## Commands
<Details>
    <summary><em>Commands</em></summary>

# Commonly used

## Create an environment

```bash=
conda create --name my_new_env python=3.8
```

## Enable environment to use by `activate`

```bash=
conda activate my_new_env
```

## Leave the enviroment

```bash=
conda deactivate my_new_env
```

## Check the total environment in the machine
```bash=
conda env list
```

## Remove Environment
```bash
conda env remove -n ENV_NAME
```

## Conda with open cv
```bash=
conda create --name cvenv python=3.8 
conda install --file requirements.txt
conda install -c conda-forge opencv
```



## Save Environment package to `requirement.txt`
- save env.yml
```bash
conda env export > environment.yml --no-builds
```
- install from env.yml
```bash
conda env create -f environment.yml
```


</Details>
