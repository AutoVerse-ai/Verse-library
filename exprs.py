from dataclasses import dataclass
from itertools import product
from pprint import pp
import re
from subprocess import PIPE, Popen
from typing import Tuple, Union

@dataclass
class ExperimentResult:
    variant: str
    max_mem: float
    ret_code: int
    num_A: int 
    type_A: str 
    Map: str 
    postCont: str 
    noisy_S: str
    node_count: Tuple[int, int]
    duration: float

# xprms = [
#     # "v" + 
#     "".join(l) for l in product("brn8", ("", "i"))]
expr_list = [
    "exp1/exp1.py",
    "exp9/exp9_dryvr.py",
    "exp9/exp9_neureach.py",
    "exp10/exp10_dryvr.py",
    "exp3/exp3.py",
    "exp2/exp2_straight.py",
    "exp2/exp2_curve.py",
    "exp5/exp5.py",
    "exp4/exp4.py",
    "exp6/exp6_dryvr.py",
    "exp6/exp6_neureach.py"
]
rslts = []
for expr in expr_list:
    cmd = Popen(f"/usr/bin/time -v -- python3.8 demo/tacas2023/{expr}", stdout=PIPE, stderr=PIPE, shell=True)
    print(f"run '{expr}', pid={cmd.pid}")
    ret = cmd.wait()
    stderr = cmd.stderr.readlines()
    stdout = cmd.stdout.readlines() 
    max_mem = float(re.search("\\d+", [l.decode("utf-8") for l in stderr if b"Maximum" in l][0]).group(0)) / 1_000
    filtered_info = [l.decode("utf-8") for l in stdout if b"#A" in l]
    if len(filtered_info) == 0:
        print(b"".join(stdout))
        print(b"".join(stderr))
        exit(2)
    info = eval(filtered_info[0])
    rslt = ExperimentResult(expr, max_mem, ret, info["#A"], info["A"], info['Map'], info['postCont'], info['Noisy S'], info['# Tr'], info['Run Time'])
    pp(rslt)
    if rslt.ret_code != 0:
        print(f"uh oh, var={expr} ret={rslt.ret_code}")
    rslts.append(rslt)

pp(rslts)

for i in range(0, len(rslts)):
    res = rslts[i]
    print(f"{res.num_A} & {res.type_A} & {res.Map} & {res.postCont} & {res.noisy_S} & {res.node_count} & {round(res.duration,2)}\\\\")
