#!/bin/bash

python main.py -s -d shesd -ds air_passengers
python main.py -s -d shesd -ds env_telemetry
python main.py -s -d shesd -ds weather_aus

python main.py -s -d ets -ds air_passengers
python main.py -s -d ets -ds env_telemetry
python main.py -s -d ets -ds weather_aus
