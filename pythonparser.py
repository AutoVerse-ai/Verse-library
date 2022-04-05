#parse python file

#REQUIRES PYTHON 3.8!
from cgitb import reset
#import clang.cindex
import typing
import json
import sys
from typing import List, Tuple
import re 
import itertools
import ast
#import pycfg
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
    
    def print(self):
        print(self.code)



class Guard(Statement):
    def __init__(self, code, mode, modeType):
        super().__init__(code, mode, modeType)

    def __init__(self, code):
        super().__init__(code)

    def isModeCheck(self):
        return self.modeType != None

    def parseGuard(node, code):
        #TODO parse out mode and modeType
        return Guard(ast.get_source_segment(code, node.test))
        

class Reset(Statement):
    def __init__(self, code, mode, modeType):
        super().__init__(code, mode, modeType)

    def __init__(self, code):
        super().__init__(code)

    def isModeUpdate(self):
        return self.modeType != None

    def parseReset(node, code):
        #TODO parse out mode and modeType
        return Reset(ast.get_source_segment(code, node))

def walktree(code, tree):
    vars = []
    out = []
    for node in ast.walk(tree): #don't think we want to walk the whole thing because lose ordering/depth
        if isinstance(node, ast.FunctionDef):
            if node.name == 'controller':
                #print(node.body)
                out = parsenodelist(code, node.body, False)
                #args = node.args
                #for arg in args:
                #        vars.append(arg.arg)
                        #todo: what to add for return values
    print(vars)
    return out

def parsenodelist(code, nodes, addResets):
    childrens_guards=[]
    childrens_resets=[]
    results = []
    found = []
    for childnode in nodes:
        if isinstance(childnode, ast.Assign) and addResets:
            reset = Reset.parseReset(childnode, code)
            print("found reset: " + reset.code)
            childrens_resets.append(reset)
        if isinstance(childnode, ast.If):
            guard = Guard.parseGuard(childnode, code)
            childrens_guards.append(guard)
            print("found if statement: " + guard.code)
            tempresults = parsenodelist(code, childnode.body, True)
            for result in tempresults:
                found.append(results)
        #TODO - can add support for elif and else
    print("********")
    print("Begin ending iteration:")
    print("We have found this many items before: " + str(len(found)))
    for item in found:
        if isinstance(item, Statement):
            print(item.code)
        else:
            print(item)
    
    print("And now we want to add these -----")
    for item in childrens_guards:
        print(item.code)
    for item in childrens_resets:
        print(item.code)
    print("-------")

    if len(found) == 0 and len(childrens_resets) > 0:
        found.append([])
    for item in found:
        for reset in childrens_resets:
        #for item in found:
            item.append(reset)
        results.append(item)
        for guard in childrens_guards:
            item.append(guard)
        
    print("now we generated these results -----")
    for result in results:
        for item in result:
            if isinstance(item, Statement):
                print(item.code)
            else:
                print(item)
                
    print("----------")
    print("********")
    return results


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
    results = walktree(code, tree)
    print("resultsssss:")
    #for result in results:
    #    for item in result:
    #        item.print()
    #    print()
    print(results)

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

