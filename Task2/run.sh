#!/bin/bash

set -e

python main.py -s -c kmeans
python main.py -s -c agglomerative
