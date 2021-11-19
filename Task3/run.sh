#!/bin/bash

set -e

python main.py -s -d arima
python main.py -s -d ets
python main.py -s -d shesd
