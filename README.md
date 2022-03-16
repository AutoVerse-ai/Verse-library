# GraphGeneration
working repo for graph generation

Run within DryVR directory

Usage:

model generation only:

python generateGraph-new.py cfile.c jsonfilewithinitialinfo.json out.json


DryVR pipeline (this may not be ready yet, still needs work with mode names):

./fullrun cfile.c jsonfilewithinitialinfo.json out.json




Example:

python generateGraph-new.py cartoy.c singlevehiclesat.json out.json #this toy example was just for looking at nesting, doesn't have an initial json

./full_run singlevehiclesat.c singlevehiclesat.json output.json

