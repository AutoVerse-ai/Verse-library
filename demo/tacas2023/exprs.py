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
    node_count: int
    duration: float
    duration2: float
    equality: bool


# xprms = [
#     # "v" +
#     "".join(l) for l in product("brn8", ("", "i"))]
expr_list = [
    "exp1/exp1.py",
    "exp9/exp9_dryvr.py",
    # "exp9/exp9_neureach.py",
    "exp10/exp10_dryvr.py",
    "exp3/exp3.py",
    "exp2/exp2_straight.py",
    "exp2/exp2_curve.py",
    "exp5/exp5.py",
    "exp4/exp4_noise.py",
    "exp6/exp6_dryvr.py",
    # "exp6/exp6_neureach.py",
    "exp12/vanderpol_demo2.py",
    "exp12/rendezvous_demo.py",
    "exp12/gearbox_demo.py",
]
rslts = []
for expr in expr_list:
    # macos cmd
    cmd = Popen(
        f"gtime --verbose python3 demo/tacas2023/{expr} l", stdout=PIPE, stderr=PIPE, shell=True
    )
    # linux cmd
    # cmd = Popen(f"/usr/bin/time -v -- python3 demo/tacas2023/{expr} v", stdout=PIPE, stderr=PIPE, shell=True)
    print(f"run '{expr}', pid={cmd.pid}")
    ret = cmd.wait()
    stderr = cmd.stderr.readlines()
    stdout = cmd.stdout.readlines()
    max_mem = (
        float(re.search("\\d+", [l.decode("utf-8") for l in stderr if b"Maximum" in l][0]).group(0))
        / 1_000
    )
    filtered_info = [l.decode("utf-8") for l in stdout if b"#A" in l]
    if len(filtered_info) == 0:
        print(b"".join(stdout))
        print(b"".join(stderr))
        exit(2)
    elif len(filtered_info) == 1:
        info = eval(filtered_info[0])
        print(info)
        rslt = ExperimentResult(
            expr,
            max_mem,
            ret,
            info["#A"],
            info["A"],
            info["Map"],
            info["postCont"],
            info["Noisy S"],
            info["# Tr"],
            info["Run Time"],
            None,
            None,
        )
        rslts.append(rslt)
    elif len(filtered_info) == 2:
        info = eval(filtered_info[0])
        info2 = eval(filtered_info[1])
        if info["# Tr"] != info2["# Tr"]:
            rslt = ExperimentResult(
                expr,
                max_mem,
                ret,
                info["#A"],
                info["A"],
                info["Map"],
                info["postCont"],
                info["Noisy S"],
                info["# Tr"],
                info["Run Time"],
                None,
                None,
            )
            rslts.append(rslt)
            rslt = ExperimentResult(
                expr,
                max_mem,
                ret,
                info2["#A"],
                info2["A"],
                info2["Map"],
                info2["postCont"],
                info2["Noisy S"],
                info2["# Tr"],
                info2["Run Time"],
                None,
                None,
            )
            rslts.append(rslt)
        else:
            equal = [l.decode("utf-8") for l in stdout if b"? True" in l]
            rslt = ExperimentResult(
                expr,
                max_mem,
                ret,
                info["#A"],
                info["A"],
                info["Map"],
                info["postCont"],
                info["Noisy S"],
                info["# Tr"],
                info["Run Time"],
                info2["Run Time"],
                len(equal) == 2,
            )
            rslts.append(rslt)
    pp(rslt)
    if ret != 0:
        print(f"uh oh, var={expr} ret={ret}")


# pp(rslts)

for res in rslts:
    if res.duration2 is None:
        print(
            f"{res.num_A} & {res.type_A} & {res.Map} & {res.postCont} & {res.noisy_S} & {res.node_count} & {round(res.duration,2)}\\\\"
        )
    else:
        print(
            f"{res.num_A} & {res.type_A} & {res.Map} & {res.postCont} & {res.noisy_S} & {res.node_count} & {round(res.duration,2)} & {round(res.duration2,2)} & {res.equality}\\\\"
        )
