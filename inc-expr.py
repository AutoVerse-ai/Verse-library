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
    duration: float
    cache_size: float
    node_count: Tuple[int, int]
    ret_code: int
    cache_hits: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]

xprms = [
    # "v" + 
    "".join(l) for l in product("brn8", ("", "i"))]
rslts = []
for xprm in xprms:
    cmd = Popen(f"/usr/bin/time -v -- python3.8 demo/tacas2023/exp11/inc-expr.py {xprm}", stdout=PIPE, stderr=PIPE, shell=True)
    print(f"run '{xprm}', pid={cmd.pid}")
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
    rslt = ExperimentResult(xprm, max_mem, info["dur"], info["cache_size"] / 1_000_000, info["node_count"], ret, info["hits"])
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

    cache_hit_rate = inc.cache_hits[0] / (inc.cache_hits[0] + inc.cache_hits[1]) if "v" not in var else (inc.cache_hits[0][0] + inc.cache_hits[1][0]) / (inc.cache_hits[0][0] + inc.cache_hits[0][1] + inc.cache_hits[1][0] + inc.cache_hits[1][1])
    print("    & " + " & ".join([name] + [str(i) for i in [inc.node_count[1], round(no.duration, 2), round(no.max_mem), round(inc.duration, 2), round(inc.max_mem), round(inc.cache_size, 2), round(cache_hit_rate * 100, 2)]]) + " \\\\")
