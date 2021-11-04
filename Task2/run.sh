#!/bin/bash

set -e

python main.py -s -c kmeans
python main.py -s -c agglomerative
python main.py -s -c db_scan
python main.py -s -c lof
