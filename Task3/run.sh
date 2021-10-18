#!/bin/bash

set -e

python main.py -s -a arima
python main.py -s -a ets
python main.py -s -a shesd
