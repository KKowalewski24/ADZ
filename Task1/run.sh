#!/bin/bash

set -e

# KNN
python main.py -c knn -s -d adwin
python main.py -c knn -s -d ddm
python main.py -c knn -s -d hddm_a
python main.py -c knn -s -d kswin
python main.py -c knn -s -d ph

# HoeffdingTreeClassifier
python main.py -c vfdt -s -d adwin
python main.py -c vfdt -s -d ddm
python main.py -c vfdt -s -d hddm_a
python main.py -c vfdt -s -d kswin
python main.py -c vfdt -s -d ph
