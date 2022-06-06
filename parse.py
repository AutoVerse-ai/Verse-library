import ast, copy
from typing import List, Dict, Optional, TypeAlias

class Scope:
    scopes: List[Dict[str, ast.AST]]
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
        ast_dump = lambda node: ast.dump(node, indent=2) if dump else ast.unparse(node)
        for scope in self.scopes:
            for k, node in scope.items():
                print(f"{k}: ", end="")
                if isinstance(node, Function):
                    print(f"Function args: {node.args} body: {ast_dump(node.body)}")
                else:
                    print(ast_dump(node))
            print("===")

class VarSubstituter(ast.NodeTransformer):
    args: Dict[str, ast.expr]
    def __init__(self, args):
        super().__init__()
        self.args = args

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.args:
            return self.args[node.id]
        self.generic_visit(node)
        return node         # XXX needed?

class Function:
    args: List[str]
    body: ast.expr
    def __init__(self, args, body):
        self.args = args
        self.body = body

    @staticmethod
    def from_func_def(fd: ast.FunctionDef, scope: Scope) -> "Function":
        args = [a.arg for a in fd.args.args]
        scope.push()
        for a in args:
            scope.set(a, ast.Name(a, ctx=ast.Load()))
        ret = None
        for node in fd.body:
            ret = proc(node, scope)
        scope.pop()
        return Function(args, ret)

    def apply(self, args: List[ast.expr]) -> ast.expr:
        ret = copy.deepcopy(self.body)
        args = {k: v for k, v in zip(self.args, args)}
        return VarSubstituter(args).visit(ret)

def proc(node: ast.AST, scope: Scope):
    match node:
        case ast.Module(body=nodes):
            for node in nodes:
                proc(node, scope)
        case ast.For(_) | ast.While(_):
            raise NotImplementedError("loops not supported")
        case ast.If(_):
            node.test = proc(node.test, scope)
            for node in node.body:      # FIXME properly handle branching
                proc(node, scope)
            for node in node.orelse:
                proc(node, scope)
        case ast.Assign(targets=targets, value=val):
            match targets:
                case [ast.Name(id=name)]:
                    scope.set(name, proc(val, scope))
                case [ast.Attribute(_)]:
                    raise NotImplementedError("assign.attr")
                case _ if len(targets) > 1:
                    raise NotImplementedError("unpacking not supported")
        case ast.Name(id=name, ctx=ast.Load()):
            return scope.lookup(name)
        case ast.FunctionDef(name):
            scope.set(name, Function.from_func_def(node, scope))
        case ast.ClassDef(_):
            pass
        case ast.UnaryOp(op, operand):
            return ast.UnaryOp(op, proc(operand, scope))
        case ast.BinOp(left, op, right):
            return ast.BinOp(proc(left, scope), op, proc(right, scope))
        case ast.BoolOp(op, vals):
            return ast.BoolOp(op, [proc(val, scope) for val in vals])
        case ast.Compare(left, op, right):
            if len(op) > 1 or len(right) > 1:
                raise NotImplementedError("too many comparisons")
            return ast.Compare(proc(left, scope), op, [proc(right[0], scope)])
        case ast.Call(fun, args):
            scope.dump()
            fun = proc(fun, scope)
            if not isinstance(fun, Function):
                raise Exception("???")
            return fun.apply([proc(a, scope) for a in args])
        case ast.List(elts):
            return ast.List([proc(e, scope) for e in elts])
        case ast.Tuple(elts):
            return ast.Tuple([proc(e, scope) for e in elts])
        case ast.Return(value=val):
            return proc(val, scope)
        case ast.Constant(value=val):
            return node         # XXX simplification?

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
