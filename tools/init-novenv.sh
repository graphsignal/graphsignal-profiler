#!/bin/bash

set -e 

pip install -U poetry
poetry env remove --all
export POETRY_VIRTUALENVS_CREATE=false 
poetry config virtualenvs.create false
poetry install
