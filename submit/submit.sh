#!/bin/bash
# rm -Rf results
docker rm "verse-$1" --force
# mkdir results
docker build --no-cache -t "verse-$1" .
docker run -d --name="verse-$1" -it "verse-$1"
docker exec "verse-$1" python ./demo/dryvr_demo/run_all.py 
docker cp "verse-$1":/Verse-Library/results.csv  ./results/results.csv
docker rm --force "verse-$1"
docker image rm --force "verse-$1"