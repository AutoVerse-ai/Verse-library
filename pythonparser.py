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

    
    def print(self):
        print(self.code)



class Guard(Statement):
    def __init__(self, code, mode, modeType):
        super().__init__(code, mode, modeType)


    def isModeCheck(self):
        return self.modeType != None

    def parseGuard(node, code):
        #assume guard is a strict comparision (modeType == mode)
        if isinstance(node.test, ast.Compare):
            if ("mode" in str(node.test.left.id)):
                modeType = str(node.test.left.id)
                #TODO: get this to work
                mode = str(node.test.comparators[0].attr)
                return Guard(ast.get_source_segment(code, node.test), mode, modeType)
        else:
            return Guard(ast.get_source_segment(code, node.test), None, None)
        

class Reset(Statement):
    def __init__(self, code, mode, modeType):
        super().__init__(code, mode, modeType)


    def isModeUpdate(self):
        return self.modeType != None

    def parseReset(node, code):
        #assume reset is modeType = newMode
        print(str(node.targets[0]))
        if ("mode" in str(node.targets[0].id)):
            modeType = node.targets[0]
            mode = node.value
            return Reset(ast.get_source_segment(code, node), mode, modeType)
        return Reset(ast.get_source_segment(code, node), None, None)

def walktree(code, tree):
    vars = []
    out = []
    mode_dict = {}
    for node in ast.walk(tree): #don't think we want to walk the whole thing because lose ordering/depth
        if isinstance(node, ast.ClassDef):
            if "Mode" in node.name:
                modeType = str(node.name)
                modes = []
                for modeValue in node.body:
                    modes.append(str(modeValue.targets[0].id))
                mode_dict[modeType] = modes
        if isinstance(node, ast.FunctionDef):
            if node.name == 'controller':
                #print(node.body)
                out = parsenodelist(code, node.body, False, [])
                #print(type(node.args))
                args = node.args.args
                for arg in args:
                    vars.append(arg.arg)
                        #todo: what to add for return values
    return [out, args, mode_dict]



def parsenodelist(code, nodes, addResets, pathsToMe):
    childrens_guards=[]
    childrens_resets=[]
    recoutput = []

    for childnode in nodes:
        if isinstance(childnode, ast.Assign) and addResets:
            reset = Reset.parseReset(childnode, code)
            #print("found reset: " + reset.code)
            childrens_resets.append(reset)
        if isinstance(childnode, ast.If):
            guard = Guard.parseGuard(childnode, code)
            childrens_guards.append(guard)
            #print("found if statement: " + guard.code)
            tempresults = parsenodelist(code, childnode.body, True, [])
            for result in tempresults:
                recoutput.append([result, guard])

    
    pathsafterme = [] 

    if len(recoutput) == 0 and len(childrens_resets) > 0:
        pathsafterme.append(childrens_resets)
    else:
        for path,ifstatement in recoutput:
            newpath = path.copy()
            newpath.extend(childrens_resets)
            newpath.append(ifstatement)
            pathsafterme.append(newpath)
            
    
    return pathsafterme


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
    results, vars, modes = walktree(code, tree)

    print("Paths found:")
    for result in results:
        for item in result:
            item.print()
            #print(item.mode)
            #print(item.modeType)
        print()

    print("Modes found: ")
    print(modes)

    output_dict.update(input_json)
    

    

   
    # #add edge, transition(guards) and resets
    #output_dict['edge'] = edges
    #output_dict['guards'] = guards
    #output_dict['resets'] = resets
    #output_dict['vertex'] = mode_list

    output_json = json.dumps(output_dict, indent=4)
    outfile = open(output_file_name, "w")
    outfile.write(output_json)
    outfile.close()

    print("wrote json to " + output_file_name)

