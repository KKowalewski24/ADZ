#!/bin/bash

set -e

python main.py -s -ws 3000 -d adwin
python main.py -s -ws 3000 -d ddm
python main.py -s -ws 3000 -d hddm_a
python main.py -s -ws 3000 -d kswin
python main.py -s -ws 3000 -d ph

python main.py -s -ws 15000 -d adwin
python main.py -s -ws 15000 -d ddm
python main.py -s -ws 15000 -d hddm_a
python main.py -s -ws 15000 -d kswin
python main.py -s -ws 15000 -d ph
