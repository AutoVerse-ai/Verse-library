from dataclasses import dataclass
from itertools import product
from pprint import pp
import re
from subprocess import PIPE, Popen

@dataclass
class ExperimentResult:
    variant: str
    max_mem: float
    duration: float
    cache_size: float
    node_count: int
    ret_code: int

xprms = ["".join(l) for l in product(
    # ("v", ""),
    "brn8", ("", "i"))]
rslts = []
for xprm in xprms:
    print(f"run '{xprm}'")
    cmd = Popen(f"/usr/bin/time -v -- python3.8 demo/vehicle/inc-expr.py {xprm}", stdout=PIPE, stderr=PIPE, shell=True)
    ret = cmd.wait()
    stderr = cmd.stderr.readlines()
    stdout = cmd.stdout.readlines() 
    max_mem = float(re.search("\\d+", [l.decode("utf-8") for l in stderr if b"Maximum" in l][0]).group(0)) / 1_000
    filtered_info = [l.decode("utf-8") for l in stdout if b"'cache_size" in l]
    if len(filtered_info) == 0:
        print(b"".join(stdout))
        print(b"".join(stderr))
        exit(2)
    info = eval(filtered_info[0])
    rslt = ExperimentResult(xprm, max_mem, info["dur"], info["cache_size"] / 1_000_000, info["node_count"], ret)
    pp(rslt)
    if rslt.ret_code != 0:
        print(f"uh oh, var={xprm} ret={rslt.ret_code}")
    rslts.append(rslt)

pp(rslts)

for i in range(0, len(rslts), 2):
    no, inc = rslts[i:i + 2]
    var = no.variant
    if "b" in var:
        name = "baseline"
    elif "r" in var:
        name = "repeat"
    elif "n" in var:
        name = "change init"
    elif "8" in var:
        name = "change ctlr"

    print(" & ".join([name] + [str(i) for i in [inc.node_count, round(no.duration, 2), round(no.max_mem), round(inc.duration, 2), round(inc.max_mem), round(inc.cache_size, 2)]]) + " \\\\")
