import ast, copy
from types import NoneType
from typing import List, Dict, Union, Optional, Any, Tuple
from dataclasses import dataclass, field, fields
from enum import Enum, auto
import dryvr_plus_plus.scene_verifier.code_parser.astunparser as astunparser
from dryvr_plus_plus.scene_verifier.utils.utils import find

debug = False

def unparse(e):
    return astunparser.unparse(e).strip("\n")

def dbg(msg, *rest):
    if not debug:
        return rest
    print(f"\x1b\x5b31m{msg}\x1b\x5bm", end="")
    for i, a in enumerate(rest[:5]):
        print(f" \x1b\x5b3{i+2}m{a}\x1b\x5bm", end="")
    if rest[5:]:
        print("", rest[5:])
    else:
        print()
    return rest

def not_ir_ast(a) -> bool:
    """Is not some type that can be used in AST substitutions"""
    return isinstance(a, ast.arg)

def fully_cond(a) -> bool:
    """Check that the values in the whole tree is based on some conditions"""
    if isinstance(a, CondVal):
        return not dbg("cv", all(len(e.cond) == 0 for e in a.elems))
    if isinstance(a, dict):
        return dbg("obj", all(fully_cond(o) for o in a.values()))
    if isinstance(a, Lambda):
        return dbg("lambda", fully_cond(a.body))
    if not_ir_ast(a):
        return True
    return False

@dataclass
class ModeDef:
    modes: List[str] = field(default_factory=list)

@dataclass
class StateDef:
    """Variable/member set needed for simulation/verification for some object"""
    cont: List[str] = field(default_factory=list)     # Continuous variables
    disc: List[str] = field(default_factory=list)     # Discrete variables
    static: List[str] = field(default_factory=list)   # Static data in object

    def all_vars(self) -> List[str]:
        return self.cont + self.disc + self.static

ScopeValue = Union[ast.AST, "CondVal", "Lambda", "Reduction", Dict[str, "ScopeValue"]]

@dataclass
class CondValCase(ast.expr):
    """A single case of a conditional value. Values in `cond` are implicitly `and`ed together"""
    cond: List[ScopeValue]
    val: ScopeValue
    _fields = ("cond", "val")

    def __reduce__(self):
        return type(self), (self.cond, self.val)

    def __eq__(self, o) -> bool:
        if o == None or len(self.cond) != len(o.cond):
            return False
        return all(ControllerIR.ir_eq(sc, oc) for sc, oc in zip(self.cond, o.cond)) and ControllerIR.ir_eq(self.val, o.val)

    def __deepcopy__(self, memo):
        cls = self.__class__ 
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v))
        return result

@dataclass
class CondVal(ast.expr):
    """A conditional value. Actual value is the combined result from all the cases"""
    elems: List[CondValCase]
    _fields = ("elems",)

    def __reduce__(self):
        return type(self), (self.elems,)

    def __deepcopy__(self, memo):
        cls = self.__class__ 
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v))
        return result

class ReductionType(Enum):
    Any = auto()
    All = auto()
    Max = auto()
    Min = auto()
    Sum = auto()

    @staticmethod
    def from_str(s: str) -> "ReductionType":
        return getattr(ReductionType, s.title())

    def __str__(self) -> str:
        return {
            ReductionType.Any: "any",
            ReductionType.All: "all",
            ReductionType.Max: "max",
            ReductionType.Min: "min",
            ReductionType.Sum: "sum",
        }[self]

@dataclass(unsafe_hash=True)
class Reduction(ast.expr):
    """A simple reduction. Must be a reduction function (see `ReductionType`) applied to a generator
    with a single clause over a iterable"""
    op: ReductionType
    expr: ast.expr
    it: str
    value: ast.AST

    def __reduce__(self):
        return type(self), (self.op, self.expr, self.it, self.value)

    def __eq__(self, o) -> bool:
        if o == None:
            return False
        return self.op == o.op and self.it == o.it and ControllerIR.ir_eq(self.expr, o.expr) and ControllerIR.ir_eq(self.value, o.value)

    def __repr__(self) -> str:
        return f"Reduction('{self.op}', expr={unparse(self.expr)}, it='{self.it}', value={unparse(self.value)})"

    def __deepcopy__(self, memo):
        cls = self.__class__ 
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v))
        return result

Reduction._fields = tuple(f.name for f in fields(Reduction))

@dataclass
class Assert:
    cond: ast.expr
    label: Optional[str]
    pre: List[ast.expr] = field(default_factory=list)

    def __eq__(self, o) -> bool:
        if o == None:
            return False
        return len(self.pre) == len(o.pre) and all(ControllerIR.ir_eq(a, b) for a, b in zip(self.pre, o.pre)) and ControllerIR.ir_eq(self.cond, o.cond)

@dataclass
class Lambda:
    """A closure. Comes from either a `lambda` or a `def`ed function"""
    args: List[Tuple[str, Optional[str]]]
    body: Optional[ast.expr]
    asserts: List[Assert]

    @staticmethod
    def from_ast(tree: Union[ast.FunctionDef, ast.Lambda], env: "Env", veri: bool) -> "Lambda":
        args = []
        for a in tree.args.args:
            if a.annotation != None:
                if isinstance(a.annotation, ast.Constant):
                    typ = a.annotation.value
                elif isinstance(a.annotation, ast.Name):
                    typ = a.annotation.id
                else:
                    raise TypeError("weird annotation?")
                args.append((a.arg, typ))
            else:
                args.append((a.arg, None))
        env.push()
        for a, typ in args:
            env.add_hole(a, typ)
        ret = None
        if isinstance(tree, ast.FunctionDef):
            for node in tree.body:
                ret = proc(node, env, veri)
        elif isinstance(tree, ast.Lambda):
            ret = proc(tree.body, env, veri)
        asserts = env.scopes[0].asserts
        # env.dump()
        env.pop()
        # assert ret != None, "Empty function"
        return Lambda(args, ret, asserts)

    @staticmethod
    def empty() -> "Lambda":
        return Lambda(args=[], body={}, asserts=[])

    def apply(self, args: List[ast.expr]) -> Tuple[List[Assert], ast.expr]:
        ret = copy.deepcopy(self.body)
        subst = ArgSubstituter({k: v for (k, _), v in zip(self.args, args)})
        ret = subst.visit(ret)
        def visit_assert(a: Assert):
            a = copy.deepcopy(a)
            pre = [subst.visit(p) for p in a.pre]
            cond = subst.visit(a.cond)
            return Assert(cond, a.label, pre)
        asserts = [visit_assert(a) for a in copy.deepcopy(self.asserts)]
        return asserts, ret

ast_dump = lambda node, dump=False: ast.dump(node) if dump else unparse(node)

@dataclass
class ScopeLevel:
    v: Dict[str, ScopeValue] = field(default_factory=dict)
    asserts: List[Assert] = field(default_factory=list)

class ArgSubstituter(ast.NodeTransformer):
    args: Dict[str, ast.expr]
    def __init__(self, args: Dict[str, ast.expr]):
        super().__init__()
        self.args = args

    def visit_arg(self, node):
        if node.arg in self.args:
            return self.args[node.arg]
        self.generic_visit(node)
        return node

@dataclass
class ControllerIR:
    controller: Optional[Lambda]
    controller_veri: Optional[Lambda]
    unsafe: Optional[Lambda]
    asserts: List[Assert]
    state_defs: Dict[str, StateDef]
    mode_defs: Dict[str, ModeDef]

    @staticmethod
    def parse(code: Optional[str] = None, fn: Optional[str] = None) -> "ControllerIR":
        print("controller parsed")
        return ControllerIR.from_envs(*Env.parse(code, fn))

    @staticmethod
    def empty() -> "ControllerIR":
        print("controller empty")
        return ControllerIR(None, None, None, [], {}, {})

    @staticmethod
    def dump(node, dump=False):
        if node == None:
            return "None"
        if isinstance(node, ast.arg):
            return f"Hole({node.arg})"
        if isinstance(node, (ModeDef, StateDef)):
            return f"<{node}>"
        if isinstance(node, Lambda):
            return f"<Lambda args: {node.args} body: {ControllerIR.dump(node.body, dump)} asserts: {ControllerIR.dump(node.asserts, dump)}>"
        if isinstance(node, CondVal):
            return f"<CondVal [{', '.join(f'{ControllerIR.dump(e.val, dump)} if {ControllerIR.dump(e.cond, dump)}' for e in node.elems)}]>"
        if isinstance(node, Assert):
            if len(node.pre) > 0:
                pre = f"[{', '.join(ControllerIR.dump(a, dump) for a in node.pre)}] => "
            else:
                pre = ""
            return f"<Assert {pre}{ControllerIR.dump(node.cond, dump)}, \"{node.label}\">"
        if isinstance(node, ast.If):
            return f"<{{{ast_dump(node, dump)}}}>"
        if isinstance(node, Reduction):
            return f"<Reduction {node.op} {ast_dump(node.expr, dump)} for {node.it} in {ast_dump(node.value, dump)}>"
        elif isinstance(node, dict):
            return "{" + ", ".join(f"{k}: {ControllerIR.dump(v, dump)}" for k, v in node.items()) + "}"
        elif isinstance(node, list):
            return f"[{', '.join(ControllerIR.dump(n, dump) for n in node)}]"
        else:
            return ast_dump(node, dump)

    @staticmethod
    def ir_eq(a: Optional[ScopeValue], b: Optional[ScopeValue]) -> bool:
        """Equality check on the "IR" nodes"""
        return ControllerIR.dump(a) == ControllerIR.dump(b)     # FIXME Proper equality checks; dump needed cuz asts are dumb

    @staticmethod
    def from_envs(env, env_veri):
        top, top_veri = env.scopes[0].v, env_veri.scopes[0].v
        assert fully_cond(top)
        if 'controller' not in top or not isinstance(top['controller'], Lambda):
            raise TypeError("can't find controller")
        # env.dump()
        # env_veri.dump()
        controller = Env.trans_args(top['controller'], False)
        controller_veri = Env.trans_args(top_veri['controller'], True)
        unsafe = Env.trans_args(top['unsafe'], False) if 'unsafe' in top else None
        assert isinstance(controller, Lambda) and isinstance(controller_veri, Lambda) and isinstance(unsafe, (Lambda, NoneType))
        print(ControllerIR.dump(controller))
        print(ControllerIR.dump(controller_veri))
        return ControllerIR(controller, controller_veri, unsafe, env.scopes[0].asserts, env.state_defs, env.mode_defs)

    def getNextModes(self) -> List[Any]:
        controller_body = self.controller_veri.body 
        paths = []
        for variable in controller_body:
            val = controller_body[variable]
            if not isinstance(val, CondVal):
                continue
            for case in val.elems:
                if len(case.cond) > 0:
                    # print(ControllerIR.dump(case.cond), f"{variable} <- {ControllerIR.dump(case.val)}")
                    paths.append((case.cond, (variable, case.val)))
        return paths 

@dataclass
class Env():
    state_defs: Dict[str, StateDef] = field(default_factory=dict)
    mode_defs: Dict[str, ModeDef] = field(default_factory=dict)
    scopes: List[ScopeLevel] = field(default_factory=lambda: [ScopeLevel()])

    @staticmethod
    def parse(code: Optional[str] = None, fn: Optional[str] = None):
        if code != None:
            if fn != None:
                root = ast.parse(code, fn)
            else:
                root = ast.parse(code)
        elif fn != None:
            with open(fn) as f:
                cont = f.read()
            root = ast.parse(cont, fn)
        else:
            raise TypeError("need at least one of `code` and `fn`")
        root_veri = copy.deepcopy(root)
        env = Env()
        proc(root, env, False)
        env_veri = Env()
        proc(root_veri, env_veri, True)
        return env, env_veri

    def push(self):
        self.scopes = [ScopeLevel()] + self.scopes

    def pop(self):
        self.scopes = self.scopes[1:]

    def lookup(self, key):
        for env in self.scopes:
            if key in env.v:
                return env.v[key]
        return None

    def set(self, key, val):
        for env in self.scopes:
            if key in env.v:
                env.v[key] = val
                return
        self.scopes[0].v[key] = val

    def add_hole(self, name: str, typ: Optional[str]):
        self.set(name, ast.arg(name, ast.Constant(typ)))

    def add_assert(self, expr, label):
        self.scopes[0].asserts.append(Assert(expr, label))

    @staticmethod
    def dump_scope(env: ScopeLevel, dump=False):
        print("+++")
        print(".asserts:")
        for a in env.asserts:
            pre = f"if {[unparse(p) for p in a.pre]}" if a.pre != None else ""
            label = ", \"{a.label}\"" if a.label != None else ""
            print(f"  {pre}assert {unparse(a.cond)}{label}")
        for k, node in env.v.items():
            print(f"{k}: {ControllerIR.dump(node, dump)}")
        print("---")

    def dump(self, dump=False):
        print("{{{")
        for env in self.scopes:
            self.dump_scope(env, dump)
        print("}}}")

    @staticmethod
    def trans_args(sv: ScopeValue, veri: bool) -> ScopeValue:
        def trans_condval(cv: CondVal):
            for i, case in enumerate(cv.elems):
                cv.elems[i].val = Env.trans_args(case.val, veri)
                for j, cond in enumerate(case.cond):
                    cv.elems[i].cond[j] = Env.trans_args(cond, veri)
            if veri:
                return cv
            else:
                ret = find(cv.elems, lambda case: len(case.cond) == 0)
                if ret == None:
                    ret = ast.Constant(None)
                for case in reversed(copy.deepcopy(cv).elems):
                    if len(case.cond) == 0:
                        ret = case.val
                    else:
                        cond = ast.BoolOp(ast.And(), case.cond) if len(case.cond) > 1 else case.cond[0]
                        ret = ast.IfExp(cond, case.val, ret)
                dbg("converted", ControllerIR.dump(cv), ControllerIR.dump(ret))
                return ret
        """Finish up parsing to turn `ast.arg` placeholders into `ast.Name`s so that the trees can be easily evaluated later"""
            # def visit_Attribute(self, node):
            #     # if isinstance(node.value, ast.Name):
            #     #     return ast.Name(f"{node.value.id}.{node.attr}", ctx=ast.Load())
            #     return node

        if isinstance(sv, dict):
            for k, v in sv.items():
                sv[k] = Env.trans_args(v, veri)
            return sv
        if isinstance(sv, CondVal):
            return trans_condval(sv)
        if isinstance(sv, ast.AST):
            class ArgTransformer(ast.NodeTransformer):
                def __init__(self, veri: bool):
                    super().__init__()
                    self.veri = veri
                def visit_arg(self, node):
                    return ast.Name(node.arg, ctx=ast.Load())
                def visit_CondVal(self, node):
                    return trans_condval(node)
            return ArgTransformer(veri).visit(sv)
        if isinstance(sv, Lambda):
            sv.body = Env.trans_args(sv.body, veri)
            dbg("assert trans bf", ControllerIR.dump(sv.asserts))
            sv.asserts = [Env.trans_args(a, veri) for a in sv.asserts]
            dbg("assert trans af", ControllerIR.dump(sv.asserts))
            return sv
        if isinstance(sv, Assert):
            sv.cond = Env.trans_args(sv.cond, veri)
            sv.pre = [Env.trans_args(p, veri) for p in sv.pre]
            return sv
        if isinstance(sv, Reduction):
            sv.expr = Env.trans_args(sv.expr, veri)
            sv.value = Env.trans_args(sv.value, veri)
            return sv

ScopeValueMap = Dict[str, ScopeValue]

def merge_if(test: ast.expr, trues: Env, falses: Env, env: Env, veri: bool):
    # `true`, `false` and `env` should have the same level
    for true, false in zip(trues.scopes, falses.scopes):
        merge_if_single(test, true.v, false.v, env, veri)
    env.scopes[0].asserts = merge_assert(test, trues.scopes[0].asserts, falses.scopes[0].asserts, env.scopes[0].asserts)

def merge_assert(test: ast.expr, trues: List[Assert], falses: List[Assert], orig: List[Assert]):
    def merge_cond(test, asserts):
        for a in asserts:
            a.pre.append(test)
        return asserts
    # dbg("assert merge", ControllerIR.dump(trues), ControllerIR.dump(falses), ControllerIR.dump(orig))
    for o in orig:
        if o in trues:
            trues.remove(o)
        if o in falses:
            falses.remove(o)
    # dbg("assert merge diff", ControllerIR.dump(trues), ControllerIR.dump(falses), ControllerIR.dump(orig))
    m_trues, m_falses = merge_cond(test, trues), merge_cond(ast.UnaryOp(ast.Not(), test), falses)
    return m_trues + m_falses + orig

def merge_if_single(test, true: ScopeValueMap, false: ScopeValueMap, scope: Union[Env, ScopeValueMap], veri: bool):
    dbg("merge if single", ControllerIR.dump(test), true.keys(), false.keys())
    def lookup(s, k):
        if isinstance(s, Env):
            return s.lookup(k)
        return s.get(k)
    def assign(s, k, v):
        if isinstance(s, Env):
            s.set(k, v)
        else:
            s[k] = v
    for var in set(true.keys()).union(set(false.keys())):
        var_true, var_false = true.get(var), false.get(var)
        if ControllerIR.ir_eq(var_true, var_false):
            continue
        if var_true != None and var_false != None:
            assert isinstance(var_true, dict) == isinstance(var_false, dict)
        dbg("merge", var, ControllerIR.dump(test), ControllerIR.dump(var_true), ControllerIR.dump(var_false))
        if isinstance(var_true, dict):
            if not isinstance(lookup(scope, var), dict):
                if lookup(scope, var) != None:
                    dbg("???", var, lookup(scope, var))
                # dbg("if.merge.obj.init")
                assign(scope, var, {})
            var_true_emp, var_false_emp, var_scope = true.get(var, {}), false.get(var, {}), lookup(scope, var)
            # dbg(isinstance(var_true_emp, dict), isinstance(var_false_emp, dict), isinstance(var_scope, dict))
            assert isinstance(var_true_emp, dict) and isinstance(var_false_emp, dict) and isinstance(var_scope, dict)
            merge_if_single(test, var_true_emp, var_false_emp, var_scope, veri)
        else:
            var_orig = lookup(scope, var)
            if_val = merge_if_val(test, var_true, var_false, var_orig)
            assign(scope, var, copy.deepcopy(if_val))
        # dbg("merged", var, ControllerIR.dump(lookup(scope, var)))

def merge_if_val(test, true: Optional[ScopeValue], false: Optional[ScopeValue], orig: Optional[ScopeValue]) -> CondVal:
    # dbg("merge val", ControllerIR.dump(test), ControllerIR.dump(true), ControllerIR.dump(false), ControllerIR.dump(orig), false == orig)
    def merge_cond(test, val):
        if isinstance(val, CondVal):
            for elem in val.elems:
                elem.cond.append(test)
            return val
        else:
            return CondVal([CondValCase([test], val)])
    def as_cv(a):
        if a == None:
            return None
        if not isinstance(a, CondVal):
            return CondVal([CondValCase([], a)])
        return a
    true, false, orig = as_cv(true), as_cv(false), as_cv(orig)
    # dbg("merge convert", ControllerIR.dump(true), ControllerIR.dump(false), ControllerIR.dump(orig))
    if orig != None:
        for orig_cve in orig.elems:
            if true != None and orig_cve in true.elems:
                true.elems.remove(orig_cve)
            if false != None and orig_cve in false.elems:
                false.elems.remove(orig_cve)

    # dbg("merge diff", ControllerIR.dump(test), ControllerIR.dump(true), ControllerIR.dump(false), ControllerIR.dump(orig))
    true_emp, false_emp = true == None or len(true.elems) == 0, false == None or len(false.elems) == 0
    if true_emp and false_emp:
        raise Exception("no need for merge?")
    elif true_emp:
        ret = merge_cond(ast.UnaryOp(ast.Not(), test), false)
    elif false_emp:
        ret = merge_cond(test, true)
    else:
        merge_true, merge_false = merge_cond(test, true), merge_cond(ast.UnaryOp(ast.Not(), test), false)
        ret = CondVal(merge_true.elems + merge_false.elems)
    if orig != None:
        return CondVal(ret.elems + orig.elems)
    return ret

def proc_assign(target: ast.AST, val, env: Env, veri: bool):
    # dbg("proc_assign", unparse(target), val)
    def proc_assign_attr(value, attr, val):
        if proc(value, env, veri) == None:
            # dbg("proc.assign.obj.init")
            proc_assign(value, {}, env, veri)
        obj = proc(value, env, veri)
        if isinstance(val, ast.AST):
            dbg("assign attr", ControllerIR.dump(obj, True), ControllerIR.dump(value, True), attr, ControllerIR.dump(val, True))
            val = proc(val, env, veri)
            dbg("assign attr val", ControllerIR.dump(val, True))
            if val != None:

                obj[attr] = val
        else:
            obj[attr] = val
    if isinstance(target, ast.Name):
        if isinstance(val, ast.AST):
            val = proc(val, env, veri)
            if val != None:
                if isinstance(val, ast.arg):
                    assert isinstance(val.annotation, ast.Constant)
                    annotation = val.annotation.s
                    if annotation in env.state_defs:
                        for attr in env.state_defs[annotation].all_vars():
                            proc_assign_attr(target, attr, ast.Attribute(val, attr, ast.Load()))
                else:
                    env.set(target.id, val)
        else:
            env.set(target.id, val)
    elif isinstance(target, ast.Attribute):
        proc_assign_attr(target.value, target.attr, val)
    else:
        raise NotImplementedError("assign.others")

def is_main_check(node: ast.If) -> bool:
    check_comps = lambda a, b: (isinstance(a, ast.Name) and a.id == "__name__"
                                and isinstance(b, ast.Constant) and b.value == "__main__")
    return (isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and (check_comps(node.test.left, node.test.comparators[0])
             or check_comps(node.test.comparators[0], node.test.left)))

START_OF_MAIN = "--start-of-main--"

# NOTE `ast.arg` used as a placeholder for idents we don't know the value of.
# This is fine as it's never used in expressions
# NOTE `veri` as a flag to create 2 versions of ASTs, one used for simulation and one for
# verification. This is needed as one form is easier to be directly evaluated while the other easier
# for verification. The differences between them means they can't be easily converted to one another
def proc(node: ast.AST, env: Env, veri: bool) -> Any:
    if isinstance(node, ast.Module):
        for node in node.body:
            if proc(node, env, veri) == START_OF_MAIN:
                break
    elif not_ir_ast(node):
        return node
    # Data massaging
    elif isinstance(node, ast.For) or isinstance(node, ast.While):
        raise NotImplementedError("loops not supported")
    elif isinstance(node, ast.If):
        if is_main_check(node):
            return START_OF_MAIN
        test = proc(node.test, env, veri)
        dbg("proc test", ControllerIR.dump(node.test, True), ControllerIR.dump(test, True))
        dbg("proc test", ControllerIR.dump(node.test), ControllerIR.dump(test))
        true_scope = copy.deepcopy(env)
        for true in node.body:
            proc(true, true_scope, veri)
        false_scope = copy.deepcopy(env)
        for false in node.orelse:
            proc(false, false_scope, veri)
        merge_if(test, true_scope, false_scope, env, veri)

    # Definition/Assignment
    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            env.add_hole(alias.name if alias.asname == None else alias.asname, None)
    elif isinstance(node, ast.Assign):
        if len(node.targets) == 1:
            proc_assign(node.targets[0], node.value, env, veri)
        else:
            raise NotImplementedError("unpacking not supported")
    elif isinstance(node, ast.Name):# and isinstance(node.ctx, ast.Load):
        return env.lookup(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        obj = proc(node.value, env, veri)
        dbg("attr", ControllerIR.dump(node), ControllerIR.dump(obj))
        # TODO since we know what the mode and state types contain we can do some typo checking
        if not_ir_ast(obj):
            if obj.arg in env.mode_defs:
                return ast.Constant(node.attr)
            attr = ast.Attribute(obj, node.attr, ctx=ast.Load())
            dbg("ret attr", ControllerIR.dump(attr, True))
            return attr
        return obj[node.attr]
    elif isinstance(node, ast.FunctionDef):
        env.set(node.name, Lambda.from_ast(node, env, veri))
    elif isinstance(node, ast.Lambda):
        return Lambda.from_ast(node, env, veri)
    elif isinstance(node, ast.ClassDef):
        def grab_names(nodes: List[ast.stmt]):
            names = []
            for node in nodes:
                if isinstance(node, ast.Assign):
                    if len(node.targets) > 1:
                        raise NotImplementedError("multiple mode/state names at once")
                    if isinstance(node.targets[0], ast.Name):
                        names.append(node.targets[0].id)
                    else:
                        raise NotImplementedError("non ident as mode/state name")
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        names.append(node.target.id)
                    else:
                        raise NotImplementedError("non ident as mode/state name")
            return names

        # NOTE we are dupping it in `state_defs`/`mode_defs` and the scopes cuz value
        if node.name.endswith("Mode"):
            mode_def = ModeDef(grab_names(node.body))
            env.mode_defs[node.name] = mode_def
        elif node.name.endswith("State"):
            names = grab_names(node.body)
            state_vars = StateDef()
            for name in names:
                if "type" == name:
                    state_vars.static.append(name)
                elif "mode" not in name:
                    state_vars.cont.append(name)
                else:
                    state_vars.disc.append(name)
            env.state_defs[node.name] = state_vars
        env.add_hole(node.name, None)
    elif isinstance(node, ast.Assert):
        cond = proc(node.test, env, veri)
        if node.msg == None:
            env.add_assert(cond, None)
        elif isinstance(node.msg, ast.Constant):
            env.add_assert(cond, node.msg.s)
        else:
            raise NotImplementedError("dynamic string in assert")

    # Expressions
    elif isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, proc(node.operand, env, veri))
    elif isinstance(node, ast.BinOp):
        return ast.BinOp(proc(node.left, env, veri), node.op, proc(node.right, env, veri))
    elif isinstance(node, ast.BoolOp):
        return ast.BoolOp(node.op, [proc(val, env, veri) for val in node.values])
    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise NotImplementedError("too many comparisons")
        return ast.Compare(proc(node.left, env, veri), node.ops, [proc(node.comparators[0], env, veri)])
    elif isinstance(node, ast.Call):
        fun = proc(node.func, env, veri)
        dbg("call fun", ControllerIR.dump(fun, True), veri)

        if isinstance(fun, Lambda):
            args = [proc(a, env, veri) for a in node.args]
            dbg("lambda args", node.args, ControllerIR.dump(args))
            asserts, ret = fun.apply(args)
            env.scopes[0].asserts.extend(asserts)
            return ret
        if isinstance(fun, ast.Attribute):
            if isinstance(fun.value, ast.arg) and fun.value.arg == "copy" and fun.attr == "deepcopy":
                if len(node.args) > 1:
                    raise ValueError("too many args to `copy.deepcopy`")
                return proc(node.args[0], env, veri)
            return node
        if isinstance(fun, ast.arg):
            if fun.arg == "copy.deepcopy":
                raise Exception("unreachable")
            else:
                ret = copy.deepcopy(node)
                ret.args = [proc(a, env, veri) for a in ret.args]
            return ret
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in ["any", "all"]:#, "max", "min", "sum"]:      # TODO
                raise NotImplementedError(f"builtin function? {name}")
            if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
                raise NotImplementedError("reduction on non-generators")
            gens = node.args[0]
            if len(gens.generators) != 1:
                raise NotImplementedError("multiple generator clauses")
            expr = gens.elt
            gen = gens.generators[0]
            target, ifs, iter = gen.target, gen.ifs, gen.iter
            if not isinstance(target, ast.Name):
                raise NotImplementedError("complex generator target")
            env.push()
            env.add_hole(target.id, None)
            if veri:
                op = ReductionType.from_str(name)
                def cond_trans(e: ast.expr, c: ast.expr) -> ast.expr:
                    if op == ReductionType.Any:
                        return ast.BoolOp(ast.And(), [e, c])
                    else:
                        return ast.BoolOp(ast.Or(), [e, ast.UnaryOp(ast.Not(), c)])
                expr = proc(expr, env, veri)
                expr = cond_trans(expr, ast.BoolOp(ast.And(), ifs)) if len(ifs) > 0 else expr
                ret = Reduction(op, expr, target.id, proc(iter, env, veri))
            else:
                dbg("gen expr", ControllerIR.dump(gens.elt, True))
                gens.elt = proc(gens.elt, env, veri)
                dbg("gen expr proced", ControllerIR.dump(gens.elt, True))
                gen.iter = proc(gen.iter, env, veri)
                gen.ifs = [proc(cond, env, veri) for cond in gen.ifs]
                # dbg("gen no veri", ControllerIR.dump(gen.iter), ControllerIR.dump(gen.ifs),
                #     ControllerIR.dump(node, True))
                ret = node
            env.pop()
            return ret
    elif isinstance(node, ast.Return):
        return proc(node.value, env, veri) if node.value != None else None
    elif isinstance(node, ast.IfExp):
        return node

    # Literals
    elif isinstance(node, ast.List):
        return ast.List([proc(e, env, veri) for e in node.elts])
    elif isinstance(node, ast.Tuple):
        return ast.Tuple([proc(e, env, veri) for e in node.elts])
    elif isinstance(node, ast.Constant):
        return node         # XXX simplification?
    else:
        print(ControllerIR.dump(node, True))
        raise NotImplementedError(str(node.__class__))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: parse.py <file.py>")
        sys.exit(1)
    fn = sys.argv[1]
    # fn = "./demo/example_controller8.py"
    # fn = "./demo/example_two_car_sign_lane_switch.py"
    e = Env.parse(fn=fn)
    e.dump()
    ir = e.to_ir()
    print(ControllerIR.dump(ir.controller.body, False))
    for a in ir.asserts:
        print(f"assert {ControllerIR.dump(a.cond, False)}, '{a.label}'")
