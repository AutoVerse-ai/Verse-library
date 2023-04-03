#!/bin/bash
rm -Rf result
docker rm verse --force
mkdir result
docker build -t verse .
docker run -d --name=verse -it verse
docker exec verse python ./demo/dryvr_demo/run_all.py 
docker cp verse:/Verse-Library/result/results.csv  ./results.csv