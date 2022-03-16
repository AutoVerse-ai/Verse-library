# GraphGeneration
working repo for graph generation

generateGraph-new.py has the code that finds paths. It creates a mode for each path through the code and any "mode" in the code is just a variable, not a named mode. 


generateGraph.py has the old code which only allows 2 levels of if statements and isn't as stable. Reads mode variable and sets vertices based on modes. Requires that an if statement checks the mode and the new mode is set within the if statement.



Run within DryVR directory

Usage:

model generation only:

python generateGraph-new.py cfile.c jsonfilewithinitialinfo.json out.json


DryVR pipeline (this may not be ready yet, still needs work with mode names):

./fullrun cfile.c jsonfilewithinitialinfo.json out.json




Example:

python generateGraph-new.py cartoy.c singlevehiclesat.json out.json #this toy example was just for looking at nesting, doesn't have an initial json

./full_run singlevehiclesat.c singlevehiclesat.json output.json

