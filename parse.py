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
    if isinstance(node, ast.Module):
        for node in node.body:
            proc(node, scope)
    elif isinstance(node, ast.For) or isinstance(node, ast.While):
        raise NotImplementedError("loops not supported")
    elif isinstance(node, ast.If):
        node.test = proc(node.test, scope)
        for node in node.body:      # FIXME properly handle branching
            proc(node, scope)
        for node in node.orelse:
            proc(node, scope)
    elif isinstance(node, ast.Assign):
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                scope.set(target.id, proc(node.value, scope))
            elif isinstance(target, ast.Attribute):
                raise NotImplementedError("assign.attr")
        else:
            raise NotImplementedError("unpacking not supported")
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return scope.lookup(node.id)
    elif isinstance(node, ast.FunctionDef):
        scope.set(node.name, Function.from_func_def(node, scope))
    elif isinstance(node, ast.ClassDef):
        pass
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
        if not isinstance(fun, Function):
            raise Exception("???")
        return fun.apply([proc(a, scope) for a in node.args])
    elif isinstance(node, ast.List):
        return ast.List([proc(e, scope) for e in node.elts])
    elif isinstance(node, ast.Tuple):
        return ast.Tuple([proc(e, scope) for e in node.elts])
    elif isinstance(node, ast.Return):
        return proc(node.value, scope)
    elif isinstance(node, ast.Constant):
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
