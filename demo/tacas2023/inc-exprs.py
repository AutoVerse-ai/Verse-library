from dataclasses import dataclass
from itertools import product
from pprint import pp
import re
from subprocess import PIPE, Popen
from typing import Tuple, Union, List
import sys 
import time 

@dataclass
class ExperimentResult:
    variant: str
    max_mem: float
    duration: Union[List[float], float]
    cache_size: Union[List[float], float]
    node_count: int
    ret_code: int
    # cache_hits: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]
    cache_hits: Union[List[Tuple[int, int]], Tuple[int, int]]

compare = len(sys.argv)>1 and 'c' in sys.argv[1]
pal = len(sys.argv)>1 and 'l' in sys.argv[1]
if len(sys.argv)>1 and 'v' in sys.argv[1]:
    if compare:
        xprms = [
        # 'vr', 
         'vri', 
        #  'vn', 
         'vni', 
        #  'v8', 
         'v8i'
         ]
    else:
        xprms = [
        'vri', 
        #  'vril', 
         'vni', 
        #  'vnil', 
         'v8i', 
        #  'v8il'
         ]        
    # xprms = [
    #     "v" + 
    #     "".join(l) for l in product("rn8", ("", "i"))]
else:
    # xprms = [
    #     "".join(l) for l in product("rn8", ("", "i"))]
    if compare:
        xprms = [
        # 'r', 
         'ri', 
        #  'n', 
         'ni', 
        #  '8', 
         '8i'
         ]
    else:
        xprms = [
        'ri', 
        #  'ril', 
         'ni', 
        #  'nil', 
         '8i', 
        #  '8il'
         ]  
rslts = []

for xprm in xprms:
    if compare:
        cmd = Popen(f"gtime --verbose -- python3 demo/tacas2023/exp11/inc-expr.py c{xprm} {'i' if 'i' in xprm else ''}l", stdout=PIPE, stderr=PIPE, shell=True)
        # cmd = Popen(f"/usr/bin/time -v -- python3 demo/tacas2023/exp11/inc-expr.py c{xprm} {'i' if 'i' in xprm else ''}l", stdout=PIPE, stderr=PIPE, shell=True)
    else:
        cmd = Popen(f"gtime --verbose -- python3 demo/tacas2023/exp11/inc-expr.py {xprm}{'l' if pal else ''}", stdout=PIPE, stderr=PIPE, shell=True)
        # cmd = Popen(f"/usr/bin/time -v -- python3 demo/tacas2023/exp11/inc-expr.py {xprm}", stdout=PIPE, stderr=PIPE, shell=True)
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
    elif len(filtered_info) == 1:    
        info = eval(filtered_info[0])
        rslt = ExperimentResult(xprm, max_mem, info["Run Time"], info["cache_size"], info["# Tr"], ret, info["cache_hit"])
        rslts.append(rslt)
    elif len(filtered_info) == 2:    
        info = eval(filtered_info[0])
        info2 = eval(filtered_info[1])
        if info['# Tr'] != info2['# Tr']:
            rslt = ExperimentResult(xprm, max_mem, info["Run Time"], info["cache_size"], info["# Tr"], ret, info["cache_hit"])
            rslts.append(rslt)
            rslt = ExperimentResult(xprm, max_mem, info2["Run Time"], info2["cache_size"], info2["# Tr"], ret, info2["cache_hit"])
            rslts.append(rslt)
        else:
            equal = [l.decode("utf-8") for l in stdout if b"? True" in l]
            rslt = ExperimentResult(xprm, max_mem, [info["Run Time"], info2["Run Time"]], [info["cache_size"], info2["cache_size"]], info["# Tr"], ret, [info["cache_hit"], info2["cache_hit"]])
            rslts.append(rslt)      
    pp(rslt)
    if rslt.ret_code != 0:
        print(f"uh oh, var={xprm} ret={rslt.ret_code}")
    

# pp(rslts)

if not compare:
    for inc in rslts:
        var = inc.variant
        if "b" in var:
            name = "baseline"
        elif "r" in var:
            name = "repeat"
        elif "n" in var:
            name = "change init"
        elif "8" in var:
            name = "change ctlr"

        cache_hit_rate = inc.cache_hits[0] / (inc.cache_hits[0] + inc.cache_hits[1])
        print("    & " + " & ".join([name] + [str(i) for i in [inc.node_count, round(inc.duration, 2), round(inc.max_mem),round(inc.cache_size, 2), round(cache_hit_rate * 100, 2)]]) + " \\\\")

else:
    for inc in rslts:
        var = inc.variant
        if "b" in var:
            name = "baseline"
        elif "r" in var:
            name = "repeat"
        elif "n" in var:
            name = "change init"
        elif "8" in var:
            name = "change ctlr"
        no_cache_hit_rate = inc.cache_hits[0][0] / (inc.cache_hits[0][0] + inc.cache_hits[0][1])
        par_cache_hit_rate = inc.cache_hits[1][0] / (inc.cache_hits[1][0] + inc.cache_hits[1][1])
        print("    & " + " & ".join([name] + [str(i) for i in [inc.node_count, round(inc.duration[0], 2), round(inc.duration[1], 2), round(inc.cache_size[0], 2), round(inc.cache_size[1], 2), round(no_cache_hit_rate * 100, 2), round(par_cache_hit_rate * 100, 2)]]) + " \\\\")


# print(">>>>>>>>>", time.time()-start_time)
