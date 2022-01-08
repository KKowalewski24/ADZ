#!/bin/bash

python main.py -s -d cmeans -ds synthetic
python main.py -s -d kmeans -ds synthetic

python main.py -s -d cmeans -ds mammography
python main.py -s -d kmeans -ds mammography

python main.py -s -d cmeans -ds http
python main.py -s -d kmeans -ds http
