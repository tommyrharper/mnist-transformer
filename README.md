# MNIST Transformer

## Setup Instructions

1. [Install conda/miniconda](https://docs.anaconda.com/miniconda/install/)
2. Create the environment from `environment.yml`
```bash
conda env create -f environment.yml
conda activate mnist-transformer
```

## How I initially created the environment

2. Create environment and export env info
```bash
conda create --name mnist-tranformer python=3.10 -y
# install all the needed tools
conda env export > full_environment.yml
conda env export --from-history > installed_environment.yml
```

2. Then manually create `environment.yml` by copying the channels from `full_environment.yml` into `installed_environment.yml`