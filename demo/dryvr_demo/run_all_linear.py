from dataclasses import dataclass
from itertools import product
from pprint import pp
import re
from subprocess import PIPE, Popen
from typing import Tuple, Union
import csv
import sys

# import os
@dataclass
class ExperimentResult:
    tool: str
    benchmark: str
    setup: str
    result: str
    time: float
    metric2: str
    metric3: str


# xprms = [
#     # "v" + 
#     "".join(l) for l in product("brn8", ("", "i"))]
expr_list = [
    "CB",
    "DTN",
    "PLA",
    "BRK",
    "spacecraft_linear_demo.py",
    "iss_demo.py",
    "gearbox_demo.py",
    "heat3d1_demo.py",
    "heat3d2_demo.py",

]
with open('results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["benchmark","instance","result","time","accuracy"])
    # spamwriter.writerow(["TRAF22"," "," 0"," ",""])
    # spamwriter.writerow(["ROBE21",   " 1"," 0",       " ",""])
    # spamwriter.writerow(["ROBE21",   " 1"," 0",       " ",""])
    # spamwriter.writerow(["ROBE21",   " 2"," 0",       " ",""])
    # spamwriter.writerow(["ROBE21",   " 2"," 0",       " ",""])
    # spamwriter.writerow(["ROBE21",   " 3"," 0",       " ",""])
    # spamwriter.writerow(["ROBE21",   " 3"," 0",       " ",""])
    # spamwriter.writerow(["CVDP23",    " "," 0",       " ",""])
    # spamwriter.writerow(["LALO20"," W001"," 0",       " ",""])
    # spamwriter.writerow(["LALO20"," W005"," 0",       " ",""])
    # spamwriter.writerow(["LALO20", " W01"," 0",       " ",""])
    # spamwriter.writerow(["LOVO21",    " "," 0",       " ",""])
    # spamwriter.writerow(["SPRE22",    " "," 0",       " ",""])

    for expr in expr_list:
        if expr == "CB" :
            spamwriter.writerow(["Beam", " ", " 0", " ", " "])
            continue
        if expr == "DTN":
            spamwriter.writerow(["Powertrain", " ", " 0", " ", " "])
            continue
        if expr == "PLA":
            spamwriter.writerow(["Platoon", " ", " 0", " ", " "])
            continue
        if expr == "BRK":
            spamwriter.writerow(["Brake", " ", " 0", " ", " "])
            continue
        # if expr == "Gear":
        #     spamwriter.writerow(["Gear", " ", " 0", " ", " "])
        #     continue
        cmd = Popen(f"python demo/dryvr_demo/{expr}", stdout=PIPE, stderr=PIPE, shell=True)
        print(f"run '{expr}', pid={cmd.pid}")
        ret = cmd.wait()
        stderr = cmd.stderr.readlines()
        stdout = cmd.stdout.readlines()
        print(stdout)
        #max_mem = float(re.search("\\d+", [l.decode("utf-8") for l in stderr if b"Maximum" in l][0]).group(0)) / 1_000
        filtered_info = [l.decode("utf-8") for l in stdout if b"tool" in l]
        if len(filtered_info) == 0:
            print(b"".join(stdout))
            print(b"".join(stderr))
            exit(2)
        for inf in filtered_info:

            info = eval(inf)
            #rslt = ExperimentResult(info["tool"], info["benchmark"], info['setup'], info['result'], info['time'], info['metric2'], info['metric3'])

            spamwriter.writerow([info["benchmark"], " "+info['setup'], " "+info['result'], " "+str(info['time'])])
        print("opps")


#for i in range(0, len(rslts)):
    #res = rslts[i]
    #print(f"{res.num_A} & {res.type_A} & {res.Map} & {res.postCont} & {res.noisy_S} & {res.node_count} & {round(res.duration,2)}\\\\")