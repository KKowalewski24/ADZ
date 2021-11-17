#!/bin/bash

python main.py -s -c kmeans -ds synthetic -ap 30 0.001
python main.py -s -c kmeans -ds synthetic -ap 15 0.001
python main.py -s -c kmeans -ds synthetic -ap 15 0.05
python main.py -s -c agglomerative -ds synthetic -ap 3 0.01
python main.py -s -c agglomerative -ds synthetic -ap 2 0.01
python main.py -s -c agglomerative -ds synthetic -ap 1 0.01
python main.py -s -c db_scan -ds synthetic -ap 0.3
python main.py -s -c db_scan -ds synthetic -ap 0.45
python main.py -s -c db_scan -ds synthetic -ap 0.6
python main.py -s -c lof -ds synthetic -ap 50
python main.py -s -c lof -ds synthetic -ap 100
python main.py -s -c lof -ds synthetic -ap 200

python main.py -s -ds http -c kmeans -ap 20 0.01
python main.py -s -ds http -c kmeans -ap 20 0.001
python main.py -s -ds http -c kmeans -ap 20 0.005
python main.py -s -ds http -c agglomerative -ap 5 0.01
python main.py -s -ds http -c agglomerative -ap 7 0.005
python main.py -s -ds http -c agglomerative -ap 10 0.005
python main.py -s -ds http -c db_scan -ap 1
python main.py -s -ds http -c db_scan -ap 2
python main.py -s -ds http -c db_scan -ap 3
python main.py -s -ds http -c lof -ap 20
python main.py -s -ds http -c lof -ap 50
python main.py -s -ds http -c lof -ap 80

python main.py -s -ds mammography -c kmeans -ap 50 0.005
python main.py -s -ds mammography -c agglomerative -ap 40 0.01
python main.py -s -ds mammography -c db_scan -ap 5
python main.py -s -ds mammography -c lof -ap 50
