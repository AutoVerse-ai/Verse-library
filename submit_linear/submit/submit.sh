#!/bin/bash
# rm -Rf results
docker rm "verse" --force
# mkdir results
docker build --no-cache -t "verse" .
docker run -d --name="verse" -it "verse"
docker exec "verse" python ./demo/dryvr_demo/run_all_linear.py
docker cp "verse":/Verse-Library/results.csv  ./results/results.csv
docker rm --force "verse"
docker image rm --force "verse"