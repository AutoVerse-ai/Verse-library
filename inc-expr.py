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
    "brn38", ("", "i"))]
rslts = []
for xprm in xprms:
    print(f"run '{xprm}'")
    cmd = Popen(f"/usr/bin/time -v -- python3.8 demo/vehicle/inc-expr.py {xprm}", stdout=PIPE, stderr=PIPE, shell=True)
    ret = cmd.wait()
    max_mem = float(re.search("\\d+", [l.decode("utf-8") for l in cmd.stderr.readlines() if b"Maximum" in l][0]).group(0)) / 1_000
    info = eval([l.decode("utf-8") for l in cmd.stdout.readlines() if b"{'cache_size" in l][0])
    rslt = ExperimentResult(xprm, max_mem, info["dur"], info["cache_size"] / 1_000_000, info["node_count"], ret)
    pp(rslt)
    if rslt.ret_code != 0:
        print(f"uh oh, var={xprm} ret={rslt.ret_code}")
    rslts.append(rslt)

pp(rslts)

for i in range(0, len(rslts), 2):
    no, inc = rslts[i * 2:i * 2 + 2]
    pp((no.variant, [no.duration, no.max_mem, inc.duration, inc.max_mem, inc.cache_size]))
