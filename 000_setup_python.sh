#!/bin/bash

# Define a function to setup the package
function setup {
  if [ "$1" = "forMac" ]; then
    # Download and install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b -p ./miniconda -f
  elif [ "$1" = "forLinux" ]; then
    # Download and install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda -f
  fi

  if [ "$1" ]; then
    # Activate miniconda environment and create virtual environment
    source miniconda/bin/activate
  fi
  which python
  # Clone repository and install required packages
  git clone ssh://git@gitlab.cern.ch:7999/lhclumi/lumi-followup.git
  python -m pip install git+https://gitlab.cern.ch/acc-co/devops/python/acc-py-pip-config.git
  python -m pip install --no-cache nxcals
  python -m pip install jupyterlab matplotlib pandas pyarrow scipy dask PyQt5 numpy
  python -m pip install ./lumi-followup/nx2pd
}

# Call the function without arguments to setup the package
#setup
# Call the function with an argument to download Python for a specific platform
setup forMac
#setup forLinux

