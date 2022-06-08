import ast, copy
from typing import List, Dict, Union, Optional, TypeAlias, Any
from dataclasses import dataclass
from enum import Enum, auto

def dbg(msg, *rest):
    print(f"\x1b\x5b31m{msg}\x1b\x5bm", end="")
    for i, a in enumerate(rest[:5]):
        print(f" \x1b\x5b3{i+2}m{a}\x1b\x5bm", end="")
    if rest[5:]:
        print("", rest[5:])
    else:
        print()

ScopeValue: TypeAlias = Union[ast.AST, "CondVal", "Lambda", Dict[str, "ScopeValue"]]

@dataclass
class CondValCase:
    """A single case of a conditional value. Values in `cond` are implicitly `and`ed together"""
    cond: List[ScopeValue]
    val: ScopeValue

    def __eq__(self, o) -> bool:
        if o == None or len(self.cond) != len(o.cond):
            return False
        return all(ir_eq(sc, oc) for sc, oc in zip(self.cond, o.cond)) and ir_eq(self.val, o.val)

@dataclass
class CondVal:
    """A conditional value. Actual value is the combined result from all the cases"""
    elems: List[CondValCase]

class ReductionType(Enum):
    Any = auto()
    All = auto()
    Max = auto()
    Min = auto()
    Sum = auto()

    @staticmethod
    def from_str(s: str) -> "ReductionType":
        return {
            "any": ReductionType.Any,
            "all": ReductionType.All,
            "max": ReductionType.Max,
            "min": ReductionType.Min,
            "sum": ReductionType.Sum,
        }[s]

@dataclass
class Reduction:
    """A simple reduction. Must be a reduction function (see `ReductionType`) applied to a generator
    with a single clause over a iterable"""
    op: ReductionType
    expr: ast.expr
    it: str
    value: ast.AST

    def __eq__(self, o) -> bool:
        if o == None:
            return False
        return self.op == o.op and self.it == o.it and ir_eq(self.expr, o.expr) and ir_eq(self.value, o.value)

@dataclass
class Lambda:
    """A closure. Comes from either a `lambda` or a `def`ed function"""
    args: List[str]
    body: ast.expr
    def __init__(self, args, body):
        self.args = args
        self.body = body

    @staticmethod
    def from_ast(tree: Union[ast.FunctionDef, ast.Lambda], scope: "Scope") -> "Lambda":
        args = [a.arg for a in tree.args.args]
        scope.push()
        for a in args:
            scope.set(a, ast.arg(a))
        ret = None
        if isinstance(tree, ast.FunctionDef):
            for node in tree.body:
                ret = proc(node, scope)
        else:
            ret = proc(tree.body, scope)
        scope.pop()
        return Lambda(args, ret)

    def apply(self, args: List[ast.expr]) -> ast.expr:
        ret = copy.deepcopy(self.body)
        return ArgSubstituter({k: v for k, v in zip(self.args, args)}).visit(ret)

ast_dump = lambda node, dump=False: ast.dump(node, indent=2) if dump else ast.unparse(node)

def ir_dump(node, dump=False):
    if node == None:
        return "None"
    if isinstance(node, Lambda):
        return f"<Lambda args: {node.args} body: {ir_dump(node.body, dump)}>"
    if isinstance(node, CondVal):
        return f"<CondVal{''.join(f' [{ir_dump(e.val, dump)} if {ir_dump(e.cond, dump)}]' for e in node.elems)}>"
    if isinstance(node, ast.If):
        return f"<{{{ast_dump(node, dump)}}}>"
    if isinstance(node, Reduction):
        return f"<Reduction {node.op} {ast_dump(node.expr, dump)} for {node.it} in {ast_dump(node.value, dump)}>"
    elif isinstance(node, dict):
        return "<Object " + " ".join(f"{k}: {ir_dump(v, dump)}" for k, v in node.items()) + ">"
    elif isinstance(node, list):
        return f"[{', '.join(ir_dump(n, dump) for n in node)}]"
    else:
        return ast_dump(node, dump)

def ir_eq(a: Optional[ScopeValue], b: Optional[ScopeValue]) -> bool:
    """Equality check on the "IR" nodes"""
    return ir_dump(a) == ir_dump(b)     # FIXME Proper equality checks; dump needed cuz asts are dumb

class Scope:
    scopes: List[Dict[str, ScopeValue]]
    def __init__(self):
        self.scopes = [{}]

    def push(self):
        self.scopes = [{}] + self.scopes

    def pop(self):
        self.scopes = self.scopes[1:]

    def lookup(self, key):
        for scope in self.scopes:
            if key in scope:
                return scope[key]
        return None

    def set(self, key, val):
        for scope in self.scopes:
            if key in scope:
                scope[key] = val
                return
        self.scopes[0][key] = val

    def dump(self, dump=False):
        for scope in self.scopes:
            for k, node in scope.items():
                print(f"{k}: {ir_dump(node, dump)}")
            print("===")

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

def merge_if(test: ast.expr, trues: Scope, falses: Scope, scope: Scope):
    # `true`, `false` and `scope` should have the same level
    for true, false in zip(trues.scopes, falses.scopes):
        merge_if_single(test, true, false, scope)

def merge_if_single(test, true: Dict[str, ScopeValue], false: Dict[str, ScopeValue], scope: Union[Scope, Dict[str, ScopeValue]]):
    dbg("merge if single", ir_dump(test), true.keys(), false.keys())
    def lookup(s, k):
        if isinstance(s, Scope):
            return s.lookup(k)
        return s.get(k)
    def assign(s, k, v):
        if isinstance(s, Scope):
            s.set(k, v)
        else:
            s[k] = v
    for var in set(true.keys()).union(set(false.keys())):
        var_true, var_false = true.get(var), false.get(var)
        if ir_eq(var_true, var_false):
            continue
        if var_true != None and var_false != None:
            assert isinstance(var_true, dict) == isinstance(var_false, dict)
        dbg("merge", var, ir_dump(test), ir_dump(var_true), ir_dump(var_false))
        if isinstance(var_true, dict):
            if not isinstance(lookup(scope, var), dict):
                if lookup(scope, var) != None:
                    dbg("???", var, lookup(scope, var))
                dbg("if.merge.obj.init")
                assign(scope, var, {})
            var_true_emp, var_false_emp, var_scope = true.get(var, {}), false.get(var, {}), lookup(scope, var)
            assert isinstance(var_true_emp, dict) and isinstance(var_false_emp, dict) and isinstance(var_scope, dict)
            merge_if_single(test, var_true_emp, var_false_emp, var_scope)
        else:
            if_val = merge_if_val(test, var_true, var_false, lookup(scope, var))
            print(ir_dump(if_val))
            assign(scope, var, if_val)

def merge_if_val(test, true: Optional[ScopeValue], false: Optional[ScopeValue], orig: Optional[ScopeValue]) -> CondVal:
    dbg("merge val", ir_dump(test), ir_dump(true), ir_dump(false), ir_dump(orig), false == orig)
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
    dbg("merge convert", ir_dump(true), ir_dump(false), ir_dump(orig))
    if orig != None:
        for orig_cve in orig.elems:
            if true != None and orig_cve in true.elems:
                true.elems.remove(orig_cve)
            if false != None and orig_cve in false.elems:
                false.elems.remove(orig_cve)

    dbg("merge diff", ir_dump(true), ir_dump(false), ir_dump(orig))
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

def proc_assign(target: ast.AST, val, scope: Scope):
    dbg("proc_assign", ast.unparse(target), val)
    if isinstance(target, ast.Name):
        if isinstance(val, ast.AST):
            scope.set(target.id, proc(val, scope))
        else:
            scope.set(target.id, val)
    elif isinstance(target, ast.Attribute):
        if proc(target.value, scope) == None:
            dbg("proc.assign.obj.init")
            proc_assign(target.value, {}, scope)
        obj = proc(target.value, scope)
        obj[target.attr] = val
    else:
        raise NotImplementedError("assign.others")

# NOTE `ast.arg` used as a placeholder for idents we don't know the value of.
# This is fine as it's never used in expressions
def proc(node: ast.AST, scope: Scope) -> Any:
    if isinstance(node, ast.Module):
        for node in node.body:
            proc(node, scope)
    elif isinstance(node, ast.arg):
        return node
    # Data massaging
    elif isinstance(node, ast.For) or isinstance(node, ast.While):
        raise NotImplementedError("loops not supported")
    elif isinstance(node, ast.If):
        test = proc(node.test, scope)
        true_scope = copy.deepcopy(scope)
        for true in node.body:
            proc(true, true_scope)
        false_scope = copy.deepcopy(scope)
        for false in node.orelse:
            proc(false, false_scope)
        merge_if(test, true_scope, false_scope, scope)

    # Definition/Assignment
    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.asname == None:
                scope.set(alias.name, ast.arg(alias.name))
            else:
                scope.set(alias.asname, ast.arg(alias.asname))
    elif isinstance(node, ast.Assign):
        if len(node.targets) == 1:
            proc_assign(node.targets[0], node.value, scope)
        else:
            raise NotImplementedError("unpacking not supported")
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return scope.lookup(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        obj = proc(node.value, scope)
        if isinstance(obj, ast.arg):
            return node
        return obj[node.attr]
    elif isinstance(node, ast.FunctionDef):
        scope.set(node.name, Lambda.from_ast(node, scope))
    elif isinstance(node, ast.Lambda):
        return Lambda.from_ast(node, scope)
    elif isinstance(node, ast.ClassDef):
        scope.set(node.name, ast.arg(node.name))

    # Expressions
    elif isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, proc(node.operand, scope))
    elif isinstance(node, ast.BinOp):
        return ast.BinOp(proc(node.left, scope), node.op, proc(node.right, scope))
    elif isinstance(node, ast.BoolOp):
        return ast.BoolOp(node.op, [proc(val, scope) for val in node.values])
    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise NotImplementedError("too many comparisons")
        return ast.Compare(proc(node.left, scope), node.ops, [proc(node.comparators[0], scope)])
    elif isinstance(node, ast.Call):
        fun = proc(node.func, scope)
        if isinstance(fun, Lambda):
            return fun.apply([proc(a, scope) for a in node.args])
        if isinstance(fun, ast.Attribute):
            ret = copy.deepcopy(node)
            ret.args = [proc(a, scope) for a in ret.args]
            return ret
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in ["any", "all"]:#, "max", "min", "sum"]:      # TODO
                raise NotImplementedError(f"builtin function? {name}")
            if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
                raise NotImplementedError("reduction on non-generators")
            gen = node.args[0]
            if len(gen.generators) != 1:
                raise NotImplementedError("multiple generator clauses")
            op = ReductionType.from_str(name)
            expr = gen.elt
            gen = gen.generators[0]
            target, ifs, iter = gen.target, gen.ifs, gen.iter
            if not isinstance(target, ast.Name):
                raise NotImplementedError("complex generator target")
            def cond_trans(e: ast.expr, c: ast.expr) -> ast.expr:
                if op == ReductionType.Any:
                    return ast.BoolOp(ast.And(), [e, c])
                else:
                    return ast.BoolOp(ast.Or, [e, ast.UnaryOp(ast.Not(), c)])
            scope.push()
            scope.set(target.id, ast.arg(target.id))
            expr = proc(expr, scope)
            scope.pop()
            expr = cond_trans(expr, ast.BoolOp(ast.And(), ifs))
            return Reduction(op, expr, target.id, proc(iter, scope))
    elif isinstance(node, ast.Return):
        return proc(node.value, scope) if node.value != None else None
    elif isinstance(node, ast.IfExp):
        return ast.If(node.test, [node.body], [node.orelse])

    # Literals
    elif isinstance(node, ast.List):
        return ast.List([proc(e, scope) for e in node.elts])
    elif isinstance(node, ast.Tuple):
        return ast.Tuple([proc(e, scope) for e in node.elts])
    elif isinstance(node, ast.Constant):
        return node         # XXX simplification?
    else:
        raise NotImplementedError(str(node.__class__))

def parse(fn: str):
    with open(fn) as f:
        cont = f.read()
    root = ast.parse(cont, fn)
    scope = Scope()
    proc(root, scope)
    scope.dump()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: parse.py <file.py>")
        sys.exit(1)
    parse(sys.argv[1])
