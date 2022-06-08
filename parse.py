import ast, copy
from typing import List, Dict, Union, Optional, TypeAlias, Any
from dataclasses import dataclass
from enum import Enum, auto

class Argument():
    pass

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
    op: ReductionType
    expr: ast.expr
    it: str
    value: ast.AST

class If:
    test: ast.expr
    true: ast.expr
    false: Optional[ast.expr]
    def __init__(self, test, true, false=None):
        self.test = test
        self.true = true
        self.false = false

@dataclass
class Lambda:
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
        for node in tree.body:
            ret = proc(node, scope)
        scope.pop()
        return Lambda(args, ret)

    def apply(self, args: List[ast.expr]) -> ast.expr:
        ret = copy.deepcopy(self.body)
        args = {k: v for k, v in zip(self.args, args)}
        return ArgSubstituter(args).visit(ret)

ast_dump = lambda node, dump=False: ast.dump(node, indent=2) if dump else ast.unparse(node)

def ir_dump(node, dump=False):
    if isinstance(node, Lambda):
        return f"<Lambda args: {node.args} body: {ir_dump(node.body, dump)}>"
    if isinstance(node, ast.If):
        return f"<{{{ast_dump(node, dump)}}}>"
    if isinstance(node, Reduction):
        return f"<Reduction {node.op}({ast_dump(node.expr, dump)} for {node.it} in {ast_dump(node.value, dump)}>"
    # if isinstance(node, If):
    #     if node.false == None:
    #         return f"<If test: {node.test} true: {ir_dump(node.true)}>"
    #     return f"<If test: {node.test} true: {ir_dump(node.true)} false: {ir_dump(node.false)}>"
    elif isinstance(node, dict):
        return "<Object " + " ".join(f"{k}: {ir_dump(v, dump)}" for k, v in node.items()) + ">"
    else:
        return ast_dump(node, dump)

ScopeValue: TypeAlias = Union[ast.AST, If, Lambda, Dict[str, "ScopeValue"]]   # TODO
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
    def __init__(self, args):
        super().__init__()
        self.args = args

    def visit_arg(self, node):
        if node.arg in self.args:
            return self.args[node.arg]
        self.generic_visit(node)
        return node         # XXX needed?

def merge_if(test, true, false, scope: Dict[str, ScopeValue]):
    for var in set(true.keys()).union(set(false.keys())):
        if true.get(var) != None and false.get(var) != None:
            assert isinstance(true.get(var), dict) == isinstance(false.get(var), dict)
        if isinstance(true.get(var), dict):
            if not isinstance(scope.get(var), dict):
                if var in scope:
                    print("???", var, scope[var])
                scope[var] = {}
            merge_if(test, true.get(var, {}), false.get(var, {}), scope[var])
        else:
            if true.get(var) == None:
                scope[var] = ast.If(ast.UnaryOp(ast.Not(), test), [false.get(var)], [])
            elif false.get(var) == None:
                scope[var] = ast.If(test, [true.get(var)], [])
            else:
                scope[var] = ast.If(test, [true.get(var)], [false.get(var)])

def proc_assign(target: ast.AST, val, scope: Scope):
    if isinstance(target, ast.Name):
        if isinstance(val, ast.AST):
            scope.set(target.id, proc(val, scope))
        else:
            scope.set(target.id, val)
    elif isinstance(target, ast.Attribute):
        if proc(target.value, scope) == None:
            proc_assign(target.value, {}, scope)
        obj = proc(target.value, scope)
        obj[target.attr] = val
    else:
        raise NotImplementedError("assign.others")

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
        true_scope.push()
        for true in node.body:
            proc(true, true_scope)
        false_scope = copy.deepcopy(scope)
        false_scope.push()
        for false in node.orelse:
            proc(false, false_scope)
        merge_if(test, true_scope.scopes[0], false_scope.scopes[0], scope.scopes[0])

    # Definition/Assignment
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
        pass

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
            if name not in ["any", "all"]:#, "max", "min", "sum"]:
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
        print(ast.dump(node))
        print(proc(node.func.value, scope))
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
    scope.dump(True)
    print(ir_dump(scope.lookup("controller").body["mode"].test))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: parse.py <file.py>")
        sys.exit(1)
    parse(sys.argv[1])
