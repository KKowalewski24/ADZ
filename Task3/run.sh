#!/bin/bash

python main.py -s -d shesd -ds air_passengers
python main.py -s -d shesd -ds alcohol_sales
python main.py -s -d shesd -ds gold_price

python main.py -s -d arima -ds air_passengers
python main.py -s -d arima -ds alcohol_sales
python main.py -s -d arima -ds gold_price
