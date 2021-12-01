#!/bin/bash

python main.py -s -ds http -d fast_abod
python main.py -s -ds mammography -d fast_abod
python main.py -s -ds synthetic -d fast_abod

python main.py -s -ds http -d bnl
python main.py -s -ds mammography -d bnl
python main.py -s -ds synthetic -d bnl
