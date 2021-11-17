#!/bin/bash

python main.py -s -c kmeans -ds synthetic -ap 15 0.05
python main.py -s -c kmeans -ds http -ap
python main.py -s -c kmeans -ds mammography -ap

python main.py -s -c agglomerative -ds http -ap
python main.py -s -c agglomerative -ds mammography -ap
python main.py -s -c agglomerative -ds synthetic -ap

python main.py -s -c db_scan -ds synthetic -ap 0.45
python main.py -s -c db_scan -ds http -ap
python main.py -s -c db_scan -ds mammography -ap

python main.py -s -c lof -ds http -ap
python main.py -s -c lof -ds mammography -ap
python main.py -s -c lof -ds synthetic -ap
