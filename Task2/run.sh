#!/bin/bash

python main.py -s -c kmeans -ds  -ap
python main.py -s -c kmeans -ds  -ap
python main.py -s -c kmeans -ds synthetic -ap

python main.py -s -c agglomerative -ds  -ap
python main.py -s -c agglomerative -ds  -ap
python main.py -s -c agglomerative -ds synthetic -ap

python main.py -s -c db_scan -ds  -ap
python main.py -s -c db_scan -ds  -ap
python main.py -s -c db_scan -ds synthetic -ap

python main.py -s -c lof -ds  -ap
python main.py -s -c lof -ds  -ap
python main.py -s -c lof -ds synthetic -ap
