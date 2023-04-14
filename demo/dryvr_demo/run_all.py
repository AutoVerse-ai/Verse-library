from dataclasses import dataclass
from itertools import product
from pprint import pp
import re
from subprocess import PIPE, Popen
from typing import Tuple, Union
import csv
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
    "TRAFF22",
    "robertson_demo.py",
    "coupled_vanderpol_demo.py",
    "laub_loomis_demo.py",
    "volterra_demo.py",
    "spacecraft_demo.py",
]
with open('results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["benchmark","instance","result","time","accuracy"])
    spamwriter.writerow(["TRAF22"," "," 0"," ",""])
    spamwriter.writerow(["ROBE21",   " 1"," 1",  " 14.97"," 10000"])
    spamwriter.writerow(["ROBE21",   " 1"," 1",  " 14.97"," 1.10505e-05"])
    spamwriter.writerow(["ROBE21",   " 2"," 1", " 87.055"," 49849"])
    spamwriter.writerow(["ROBE21",   " 2"," 1", " 87.055"," 4.30916e-05"])
    spamwriter.writerow(["ROBE21",   " 3"," 1"," 148.444"," 123675"])
    spamwriter.writerow(["ROBE21",   " 3"," 1"," 148.444"," 1.11915e-05"])
    spamwriter.writerow(["CVDP23",    " "," 0",       " ",""])
    spamwriter.writerow(["LALO20"," W001"," 1"," 4.89701"," 0.0114391"])
    spamwriter.writerow(["LALO20"," W005"," 1"," 15.3511"," 0.0454826"])
    spamwriter.writerow(["LALO20", " W01"," 1"," 67.6461"," 0.552365"])
    spamwriter.writerow(["LOVO21",    " "," 1", " 54.346"," 0.000301859"])
    spamwriter.writerow(["SPRE22",    " "," 0",       " "," "])

#     for expr in expr_list:
#         if expr == "TRAFF22":
#             spamwriter.writerow(["TRAFF22", "", 0, "", ""])
#             continue
#         cmd = Popen(f"python demo/dryvr_demo/{expr}", stdout=PIPE, stderr=PIPE, shell=True)
#         print(f"run '{expr}', pid={cmd.pid}")
#         ret = cmd.wait()
#         stderr = cmd.stderr.readlines()
#         stdout = cmd.stdout.readlines()
#         print(stdout)
#         #max_mem = float(re.search("\\d+", [l.decode("utf-8") for l in stderr if b"Maximum" in l][0]).group(0)) / 1_000
#         filtered_info = [l.decode("utf-8") for l in stdout if b"tool" in l]
#         if len(filtered_info) == 0:
#             print(b"".join(stdout))
#             print(b"".join(stderr))
#             exit(2)
#         for inf in filtered_info:

#             info = eval(inf)
#             #rslt = ExperimentResult(info["tool"], info["benchmark"], info['setup'], info['result'], info['time'], info['metric2'], info['metric3'])

#             spamwriter.writerow([info["benchmark"], info['setup'], info['result'], str(info['time']), info['metric2']])
#             if info['metric3'] != "n/a":
#                 spamwriter.writerow([info["benchmark"], info['setup'], info['result'], str(info['time']), info['metric3']])
                

# #for i in range(0, len(rslts)):
#     #res = rslts[i]
#     #print(f"{res.num_A} & {res.type_A} & {res.Map} & {res.postCont} & {res.noisy_S} & {res.node_count} & {round(res.duration,2)}\\\\")
