#!/bin/bash

set -e

# KNN
python main.py -c knn -s -d ddm
python main.py -c knn -s -d eddm
python main.py -c knn -s -d adwin
python main.py -c knn -s -d kswin
python main.py -c knn -s -d ph

# VFDT - HoeffdingTreeClassifier
python main.py -c vfdt -s -d ddm
python main.py -c vfdt -s -d eddm
python main.py -c vfdt -s -d adwin
python main.py -c vfdt -s -d kswin
python main.py -c vfdt -s -d ph
