#parse python file

from cgitb import reset
import clang.cindex
import typing
import json
import sys
from typing import List, Tuple
import re 
import itertools
import ast
import pycfg
#from cfg import CFG
#import clang.cfg

class Statement:
  def __init__(self, code, mode, modeType):
    self.code = code
    self.modeType = modeType
    self.mode = mode

    def __init__(self, code):
        self.code = code
        self.modeType = None
        self.mode = None

class Guard(Statement):
    def __init__(self, code, mode, modeType):
        super().__init__(self, code, mode, modeType)

    def __init__(self, code):
        super().__init__(self, code)

    def isModeCheck():
        return modeType != None

class Reset(Statement):
    def __init__(self, code, mode, modeType):
        super().__init__(self, code, mode, modeType)

    def __init__(self, code):
        super().__init__(self, code)

    def isModeUpdate():
        return modeType != None

def walktree(code, tree):
    vars = []
    for node in ast.walk(tree): #don't think we want to walk the whole thing because lose ordering/depth
        if isinstance(node, ast.FunctionDef):
            if node.name == 'controller':
                print(node.body)
                parsenodelist(code, node.body)
                #args = node.args
                #for arg in args:
                #        vars.append(arg.arg)
                        #todo: what to add for return values
    print(vars)
    return vars

def parsenodelist(code, nodes):
    childrens_guards=[]
    childrens_resets=[]
    results = []
    found = []
    for childnode in nodes:
        if isinstance(childnode, ast.Assign):
            #found.append(str(childnode.targets[0].id)+'=' + str(childnode.value.id)) #TODO, does this work with a value?
            print("found reset" + str(childnode.targets[0].id)+'=' + str(childnode.value.id))
            childrens_resets.append(str(childnode.targets[0].id)+'=' + str(childnode.value.id))
        if isinstance(childnode, ast.If):
            childrens_guards.append(ast.get_source_segment(code, childnode.test))
            #childrens_guards.append(childnode.test.id)
            print("found if statement: " + ast.get_source_segment(code, childnode.test))
            results = parsenodelist(code, childnode.body)
            for result in results:
                found.append(results)
        #TODO - can add support for elif and else
        print(type(childnode))

    if len(found) == 0 and len(childrens_resets) > 0:
        found.append([])
    for item in found:
        for reset in childrens_resets:
        #for item in found:
            item.append([reset, 'r'])
        results.append(item)
        for guard in childrens_guards:
            item.append([guard, 'g'])
    return results

def parsenodes(innode):
    for node in innode:
        #print(node)
        if isinstance(node, ast.If):
            print("found if")
            print(node.body)
        #walktree(node.body)
        #walktree(node)
            #print(node.name)

class BinOpVisitor(ast.NodeVisitor):

    def visit_BinOp(self, node):
        print(f"found BinOp at line: {node.lineno}")
        self.generic_visit(node)

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {'if': []}

    def visit_If(self, node):
        # Add the "s" attribute access here
        print("If:", node.test.left)
        self.stats["if"].append(node.body)
        self.generic_visit(node)

    def report(self):
        pprint(self.stats)




##main code###
#print(sys.argv)
if __name__ == "__main__":
    #if len(sys.argv) < 4:
    #    print("incorrect usage. call createGraph.py program inputfile outputfilename")
    #    quit()

    input_code_name = 'toythermomini.py' #sys.argv[1]
    input_file_name = 'billiard_input.json' #sys.argv[2] 
    output_file_name = 'out.json' #sys.argv[3]



    with open(input_file_name) as in_json_file:
        input_json = json.load(in_json_file)

    output_dict = {
    }

    f = open(input_code_name,'r')
    code = f.read()
    tree = ast.parse(code)
    #tree = ast.parse()
    walktree(code, tree)

    #a = Analyzer()
    #a.visit(tree)
    
    #print(ast.dump(tree, indent=4))

    #cfge = pycfg.PyCFGExtractor()

    
    #for node in ast.walk(tree):
    #    print(node.__class__.__name__)

    #print("=== full AST ===")
    #print(ast.dump(tree))

    #print()
    #print("=== visit ===")
    #vis = BinOpVisitor()
    #vis.visit(tree)

    output_dict.update(input_json)

    #file parsing set up
   
    # #add edge, transition(guards) and resets
    #output_dict['edge'] = edges
    #output_dict['guards'] = guards
    #output_dict['resets'] = resets
    #output_dict['vertex'] = mode_list

    output_json = json.dumps(output_dict, indent=4)
    #print(output_json)
    outfile = open(output_file_name, "w")
    outfile.write(output_json)
    outfile.close()

    # print("wrote json to " + output_file_name)

