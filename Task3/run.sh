#!/bin/bash

python main.py -s -d shesd -ds air_passengers
python main.py -s -d shesd -ds alcohol_sales
python main.py -s -d shesd -ds gold_price

python main.py -s -d ets -ds air_passengers
python main.py -s -d ets -ds alcohol_sales
python main.py -s -d ets -ds gold_price
