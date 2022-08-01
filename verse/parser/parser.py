import ast, copy, warnings
from typing import List, Dict, Union, Optional, Any, Tuple
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from verse.parser import astunparser
from verse.analysis.utils import find

def merge_conds(c):
    if len(c) == 0:
        return ast.Constant(True)
    if len(c) == 1:
        return c[0]
    else:
        return ast.BoolOp(ast.And(), c)

def compile_expr(e):
    return compile(ast.fix_missing_locations(ast.Expression(e)), "", "eval")

def unparse(e):
    return astunparser.unparse(e).strip("\n")

def not_ir_ast(a) -> bool:
    """Is not some type that can be used in AST substitutions"""
    return isinstance(a, ast.arg)

def fully_cond(a) -> bool:
    """Check that the values in the whole tree is based on some conditions"""
    if isinstance(a, CondVal):
        return not all(len(e.cond) == 0 for e in a.elems)
    if isinstance(a, dict):
        return all(fully_cond(o) for o in a.values())
    if isinstance(a, Lambda):
        return a.body == None or fully_cond(a.body)
    return not_ir_ast(a)

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

class CustomIR(ast.expr):
    def __reduce__(self):
        return type(self), tuple(getattr(self, k) for k in self.__class__._fields)

    def __deepcopy__(self, _memo):
        cls = self.__class__ 
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v))
        return result

    @staticmethod
    def set_fields(klass):
        klass._fields = tuple(f.name for f in fields(klass))

@dataclass
class CondValCase(CustomIR):
    """A single case of a conditional value. Values in `cond` are implicitly `and`ed together"""
    cond: List[ScopeValue]
    val: ScopeValue

    def __eq__(self, o) -> bool:
        if o == None or len(self.cond) != len(o.cond):
            return False
        return all(ControllerIR.ir_eq(sc, oc) for sc, oc in zip(self.cond, o.cond)) and ControllerIR.ir_eq(self.val, o.val)
CustomIR.set_fields(CondValCase)

@dataclass
class CondVal(CustomIR):
    """A conditional value. Actual value is the combined result from all the cases"""
    elems: List[CondValCase]
CustomIR.set_fields(CondVal)

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
class Reduction(CustomIR):
    """A simple reduction. Must be a reduction function (see `ReductionType`) applied to a generator
    with a single clause over a iterable"""
    op: ReductionType
    expr: ast.expr
    it: str
    value: ast.AST

    def __eq__(self, o) -> bool:
        if o == None:
            return False
        return self.op == o.op and self.it == o.it and ControllerIR.ir_eq(self.expr, o.expr) and ControllerIR.ir_eq(self.value, o.value)

    def __repr__(self) -> str:
        return f"Reduction('{self.op}', expr={unparse(self.expr)}, it='{self.it}', value={unparse(self.value)})"
CustomIR.set_fields(Reduction)

@dataclass
class _Assert:
    cond: ast.expr
    label: Optional[str]
    pre: List[ast.expr] = field(default_factory=list)

    def __eq__(self, o) -> bool:
        if o == None:
            return False
        return len(self.pre) == len(o.pre) and all(ControllerIR.ir_eq(a, b) for a, b in zip(self.pre, o.pre)) and ControllerIR.ir_eq(self.cond, o.cond)

@dataclass
class CompiledAssert:
    cond: Any   # FIXME type for compiled python (`code`?)
    label: str
    pre: Any

@dataclass
class Assert:
    cond: ast.expr
    label: str
    pre: ast.expr

@dataclass
class LambdaArg:
    name: str
    typ: Optional[str]
    is_list: bool

LambdaArgs = List[LambdaArg]

@dataclass
class Lambda:
    """A closure. Comes from either a `lambda` or a `def`ed function"""
    args: LambdaArgs
    body: Optional[ast.expr]
    asserts: List[_Assert]

    @staticmethod
    def from_ast(tree: Union[ast.FunctionDef, ast.Lambda], env: "Env") -> "Lambda":
        args = []
        for a in tree.args.args:
            if a.annotation != None:
                def handle_simple_ann(a):
                    if isinstance(a, ast.Constant):
                        return a.value
                    elif isinstance(a, ast.Name):
                        return a.id
                    elif isinstance(a, ast.Index):
                        return a.value.id
                    else:
                        raise TypeError(f"weird annotation? {a}")
                if isinstance(a.annotation, ast.Subscript) \
                    and isinstance(a.annotation.value, ast.Name) \
                        and a.annotation.value.id == 'List':
                    typ = handle_simple_ann(a.annotation.slice)
                    is_list = True
                else:
                    typ = handle_simple_ann(a.annotation)
                    is_list = False
                args.append(LambdaArg(a.arg, typ, is_list))
            else:
                args.append(LambdaArg(a.arg, None, False))
        env.push()
        for a in args:
            env.add_hole(a.name, a.typ)
        ret = None
        if isinstance(tree, ast.FunctionDef):
            for node in tree.body:
                ret = proc(node, env)
        elif isinstance(tree, ast.Lambda):
            ret = proc(tree.body, env)
        asserts = env.scopes[0].asserts
        env.pop()
        return Lambda(args, ret, asserts)

    @staticmethod
    def empty() -> "Lambda":
        return Lambda(args=[], body=None, asserts=[])

    def apply(self, args: List[ast.expr]) -> Tuple[List[_Assert], ast.expr]:
        ret = copy.deepcopy(self.body)
        subst = ArgSubstituter({a.name: v for a, v in zip(self.args, args)})
        ret = subst.visit(ret)
        def visit_assert(a: _Assert):
            a = copy.deepcopy(a)
            pre = [subst.visit(p) for p in a.pre]
            cond = subst.visit(a.cond)
            return _Assert(cond, a.label, pre)
        asserts = [visit_assert(a) for a in copy.deepcopy(self.asserts)]
        return asserts, ret

ast_dump = lambda node, dump=False: ast.dump(node) if dump else unparse(node)

@dataclass
class ScopeLevel:
    v: Dict[str, ScopeValue] = field(default_factory=dict)
    asserts: List[_Assert] = field(default_factory=list)

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
class ModePath:
    cond: Any
    cond_veri: ast.expr
    var: str
    val: Any
    val_veri: ast.expr

@dataclass
class ControllerIR:
    args: LambdaArgs
    paths: List[ModePath]
    asserts: List[CompiledAssert]
    asserts_veri: List[Assert]
    state_defs: Dict[str, StateDef]
    mode_defs: Dict[str, ModeDef]

    @staticmethod
    def parse(code: Optional[str] = None, fn: Optional[str] = None) -> "ControllerIR":
        return ControllerIR.from_env(Env.parse(code, fn))

    @staticmethod
    def empty() -> "ControllerIR":
        return ControllerIR([], [], [], [], {}, {})

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
        if isinstance(node, _Assert):
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
    def from_env(env):
        top = env.scopes[0].v
        if 'controller' not in top or not isinstance(top['controller'], Lambda):
            raise TypeError("can't find controller")
        controller = top['controller']
        asserts = [(a.cond, a.label if a.label != None else f"<assert {i}>", merge_conds(a.pre)) for i, a in enumerate(controller.asserts)]
        asserts_veri = [Assert(Env.trans_args(copy.deepcopy(c), True), l, Env.trans_args(copy.deepcopy(p), True)) for c, l, p in asserts]
        for a in asserts_veri:
            print(ControllerIR.dump(a.pre), ControllerIR.dump(a.cond, True))
        asserts_sim = [CompiledAssert(compile_expr(Env.trans_args(c, False)), l, compile_expr(Env.trans_args(p, False))) for c, l, p in asserts]

        assert isinstance(controller, Lambda)
        paths = []
        if not isinstance(controller.body, dict):
            raise NotImplementedError("non-object return")
        for var, val in controller.body.items():
            if not isinstance(val, CondVal):
                continue
            for case in val.elems:
                if len(case.cond) > 0:
                    cond = merge_conds(case.cond)
                    cond_veri = Env.trans_args(copy.deepcopy(cond), True)
                    val_veri = Env.trans_args(copy.deepcopy(case.val), True)
                    cond = compile_expr(Env.trans_args(cond, False))
                    val = compile_expr(Env.trans_args(case.val, False))
                    paths.append(ModePath(cond, cond_veri, var, val, val_veri))

        return ControllerIR(controller.args, paths, asserts_sim, asserts_veri, env.state_defs, env.mode_defs)

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
        env = Env()
        proc(root, env)
        return env

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
        self.set(name, ast.arg(name, ast.Constant(typ, None)))

    def add_assert(self, expr, label):
        self.scopes[0].asserts.append(_Assert(expr, label))

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
        def trans_condval(cv: CondVal, veri: bool):
            raise NotImplementedError("flatten CondVal assignments")
            for i, case in enumerate(cv.elems):
                cv.elems[i].val = Env.trans_args(case.val, veri)
                for j, cond in enumerate(case.cond):
                    cv.elems[i].cond[j] = Env.trans_args(cond, veri)
            if veri:
                return cv
            else:
                ret = find(cv.elems, lambda case: len(case.cond) == 0)
                if ret == None:
                    ret = ast.Constant(None, None)
                for case in reversed(copy.deepcopy(cv).elems):
                    if len(case.cond) == 0:
                        ret = case.val
                    else:
                        cond = ast.BoolOp(ast.And(), case.cond) if len(case.cond) > 1 else case.cond[0]
                        ret = ast.IfExp(cond, case.val, ret)
                return ret
        def trans_reduction(red: Reduction, veri: bool):
            if veri:
                red.expr = Env.trans_args(red.expr, True)
                red.value = Env.trans_args(red.value, True)
                return red
            expr = Env.trans_args(red.expr, False)
            value = Env.trans_args(red.value, False)
            gen_expr = ast.GeneratorExp(expr, [ast.comprehension(ast.Name(red.it, ctx=ast.Store()), value, [], False)])
            return ast.Call(ast.Name(str(red.op), ctx=ast.Load()), [gen_expr], [])

        if isinstance(sv, Reduction):
            return trans_reduction(sv, veri)
        if isinstance(sv, CondVal):
            return trans_condval(sv, veri)
        if isinstance(sv, ast.AST):
            class ArgTransformer(ast.NodeTransformer):
                def __init__(self, veri: bool):
                    super().__init__()
                    self.veri = veri
                def visit_arg(self, node):
                    return ast.Name(node.arg, ctx=ast.Load())
                def visit_CondVal(self, node):
                    return trans_condval(node, self.veri)
                def visit_Reduction(self, node):
                    return trans_reduction(node, self.veri)
                # def visit_Attribute(self, node: ast.Attribute) -> Any:
                #     if self.veri:
                #         value = super().visit(node.value)
                #         if isinstance(value, ast.Name):
                #             return ast.Name(f"{value.id}.{node.attr}", ctx=ast.Load())
                #         raise ValueError(f"value of attribute node is not name?: {unparse(node)}")
                #     else:
                #         return super().generic_visit(node)
            return ArgTransformer(veri).visit(sv)
        if isinstance(sv, dict):
            for k, v in sv.items():
                sv[k] = Env.trans_args(v, veri)
            return sv
        if isinstance(sv, Lambda):
            sv.body = Env.trans_args(sv.body, veri)
            sv.asserts = [Env.trans_args(a, veri) for a in sv.asserts]
            return sv
        if isinstance(sv, _Assert):
            sv.cond = Env.trans_args(sv.cond, veri)
            sv.pre = [Env.trans_args(p, veri) for p in sv.pre]
            return sv
        if isinstance(sv, Reduction):
            sv.expr = Env.trans_args(sv.expr, veri)
            sv.value = Env.trans_args(sv.value, veri)
            return sv
        print(ControllerIR.dump(sv, True))
        raise NotImplementedError(str(sv.__class__))

ScopeValueMap = Dict[str, ScopeValue]

def merge_if(test: ast.expr, trues: Env, falses: Env, env: Env):
    # `true`, `false` and `env` should have the same level
    for true, false in zip(trues.scopes, falses.scopes):
        merge_if_single(test, true.v, false.v, env)
    env.scopes[0].asserts = merge_assert(test, trues.scopes[0].asserts, falses.scopes[0].asserts, env.scopes[0].asserts)

def merge_assert(test: ast.expr, trues: List[_Assert], falses: List[_Assert], orig: List[_Assert]):
    def merge_cond(test, asserts):
        for a in asserts:
            a.pre.append(test)
        return asserts
    for o in orig:
        if o in trues:
            trues.remove(o)
        if o in falses:
            falses.remove(o)
    m_trues, m_falses = merge_cond(test, trues), merge_cond(ast.UnaryOp(ast.Not(), test), falses)
    return m_trues + m_falses + orig

def merge_if_single(test, true: ScopeValueMap, false: ScopeValueMap, scope: Union[Env, ScopeValueMap]):
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
        if isinstance(var_true, dict):
            if not isinstance(lookup(scope, var), dict):
                if lookup(scope, var) != None:
                    raise NotImplementedError(f"uncaught object assignment before attribute assignment: {var, lookup(scope, var)}")
                assign(scope, var, {})
            var_true_emp, var_false_emp, var_scope = true.get(var, {}), false.get(var, {}), lookup(scope, var)
            assert isinstance(var_true_emp, dict) and isinstance(var_false_emp, dict) and isinstance(var_scope, dict)
            merge_if_single(test, var_true_emp, var_false_emp, var_scope)
        else:
            var_orig = lookup(scope, var)
            if_val = merge_if_val(test, var_true, var_false, var_orig)
            assign(scope, var, copy.deepcopy(if_val))

def merge_if_val(test, true: Optional[ScopeValue], false: Optional[ScopeValue], orig: Optional[ScopeValue]) -> CondVal:
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
    if orig != None:
        for orig_cve in orig.elems:
            if true != None and orig_cve in true.elems:
                true.elems.remove(orig_cve)
            if false != None and orig_cve in false.elems:
                false.elems.remove(orig_cve)

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

def proc_assign(target: ast.AST, val, env: Env):
    def proc_assign_attr(value, attr, val):
        if proc(value, env) == None:
            proc_assign(value, {}, env)
        obj = proc(value, env)
        if isinstance(val, ast.AST):
            val = proc(val, env)
            if val != None:
                obj[attr] = val
        else:
            obj[attr] = val
    if isinstance(target, ast.Name):
        if isinstance(val, ast.AST):
            val = proc(val, env)
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
def proc(node: ast.AST, env: Env) -> Any:
    if isinstance(node, ast.Module):
        for node in node.body:
            if proc(node, env) == START_OF_MAIN:
                break
    elif not_ir_ast(node):
        return node
    # Data massaging
    elif isinstance(node, ast.For) or isinstance(node, ast.While):
        raise NotImplementedError("loops not supported")
    elif isinstance(node, ast.If):
        if is_main_check(node):
            return START_OF_MAIN
        test = proc(node.test, env)
        true_scope = copy.deepcopy(env)
        for true in node.body:
            proc(true, true_scope)
        false_scope = copy.deepcopy(env)
        for false in node.orelse:
            proc(false, false_scope)
        merge_if(test, true_scope, false_scope, env)

    # Definition/Assignment
    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            env.add_hole(alias.name if alias.asname == None else alias.asname, None)
    elif isinstance(node, ast.Assign):
        if len(node.targets) == 1:
            proc_assign(node.targets[0], node.value, env)
        else:
            raise NotImplementedError("unpacking not supported")
    elif isinstance(node, ast.Name):# and isinstance(node.ctx, ast.Load):
        return env.lookup(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        obj = proc(node.value, env)
        # TODO since we know what the mode and state types contain we can do some typo checking
        if not_ir_ast(obj):
            if obj.arg in env.mode_defs:
                return ast.Constant(node.attr, kind=None)
            attr = ast.Attribute(obj, node.attr, ctx=ast.Load())
            return attr
        return obj[node.attr]
    elif isinstance(node, ast.FunctionDef):
        env.set(node.name, Lambda.from_ast(node, env))
    elif isinstance(node, ast.Lambda):
        return Lambda.from_ast(node, env)
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
        cond = proc(node.test, env)
        if node.msg == None:
            env.add_assert(cond, None)
        elif isinstance(node.msg, ast.Constant):
            env.add_assert(cond, node.msg.s)
        else:
            raise NotImplementedError("dynamic string in assert")

    # Expressions
    elif isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, proc(node.operand, env))
    elif isinstance(node, ast.BinOp):
        return ast.BinOp(proc(node.left, env), node.op, proc(node.right, env))
    elif isinstance(node, ast.BoolOp):
        return ast.BoolOp(node.op, [proc(val, env) for val in node.values])
    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise NotImplementedError("too many comparisons")
        return ast.Compare(proc(node.left, env), node.ops, [proc(node.comparators[0], env)])
    elif isinstance(node, ast.Call):
        fun = proc(node.func, env)

        if isinstance(fun, Lambda):
            args = [proc(a, env) for a in node.args]
            asserts, ret = fun.apply(args)
            env.scopes[0].asserts.extend(asserts)
            return ret
        if isinstance(fun, ast.Attribute):
            if isinstance(fun.value, ast.arg) and fun.value.arg == "copy" and fun.attr == "deepcopy":
                if len(node.args) > 1:
                    raise ValueError("too many args to `copy.deepcopy`")
                return proc(node.args[0], env)
            return node
        if isinstance(fun, ast.arg):
            if fun.arg == "copy.deepcopy":
                raise Exception("unreachable")
            else:
                ret = copy.deepcopy(node)
                ret.args = [proc(a, env) for a in ret.args]
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
            op = ReductionType.from_str(name)
            def cond_trans(e: ast.expr, c: ast.expr) -> ast.expr:
                if op == ReductionType.Any:
                    return ast.BoolOp(ast.And(), [e, c])
                else:
                    return ast.BoolOp(ast.Or(), [e, ast.UnaryOp(ast.Not(), c)])
            expr = proc(expr, env)
            expr = cond_trans(expr, ast.BoolOp(ast.And(), ifs)) if len(ifs) > 0 else expr
            ret = Reduction(op, expr, target.id, proc(iter, env))
            env.pop()
            return ret
    elif isinstance(node, ast.Return):
        return proc(node.value, env) if node.value != None else None
    elif isinstance(node, ast.IfExp):
        return node
    elif isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            warnings.warn(f"Effects of this call will not be included in the result: \"{unparse(node.value)}\"")
        return None

    # Literals
    elif isinstance(node, ast.List):
        return ast.List([proc(e, env) for e in node.elts])
    elif isinstance(node, ast.Tuple):
        return ast.Tuple([proc(e, env) for e in node.elts])
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
    e = Env.parse(fn=fn)
    e.dump()
    ir = e.to_ir()
    print(ControllerIR.dump(ir.controller.body, False))
    for a in ir.asserts:
        print(f"assert {ControllerIR.dump(a.cond, False)}, '{a.label}'")
