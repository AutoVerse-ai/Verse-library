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

class Edge:
    def __init__(self, source, dest, guards, resets):
        self.source = source
        self.dest = dest
        self.guards = guards
        self.resets = resets

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
            if isinstance(node.test.comparators[0], ast.Attribute):
                if ("Mode" in str(node.test.comparators[0].value.id)):
                    modeType = str(node.test.comparators[0].value.id)
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
        if isinstance(node.value, ast.Attribute):
            #print("resets " + str(node.value.value.id))
            #print("resets " + str(node.value.attr))
            if ("Mode" in str(node.value.value.id)):
                modeType = str(node.value.value.id)
                mode = str(node.value.attr)
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

def resetString(resets):
    outstr = ""
    for reset in resets:
        outstr+= reset.code + ";"
    outstr = outstr.strip(";")
    return outstr

def guardString(guards):
    return guards

#modes are the list of all modes in the current vertex
#vertices are all the vertexs
def getIndex(modes, vertices):
    #TODO: will this work if ordering is lost, will ordering be lost?
    return vertices.index(tuple(modes))
    #for index in range(0, len(vertices)):
    #    allMatch = True
    #    for mode in modes:
    #        if not (mode in vertices[index]):
    #            allMatch = False
    #    if allMatch:
    #        return index
    return -1

def createTransition(path, vertices, modes):
    guards = []
    resets = []
    modeChecks = []
    modeUpdates = []
    for item in path:
        if isinstance(item, Guard):
            if not item.isModeCheck():
                guards.append(item)
            else:
                modeChecks.append(item)
        if isinstance(item, Reset):
            if not item.isModeUpdate():
                resets.append(item)
            else:
                modeUpdates.append(item)
    unfoundSourceModeTypes = []
    sourceModes = []
    unfoundDestModeTypes = []
    destModes = []
    for modeType in modes.keys():
        foundMode = False
        for condition in modeChecks:
            #print(condition.modeType)
            #print(modeType)
            if condition.modeType == modeType:
                sourceModes.append(condition.mode)
                foundMode = True
        if foundMode == False:
            unfoundSourceModeTypes.append(modeType)
        foundMode = False
        for condition in modeUpdates:
            if condition.modeType == modeType:
                destModes.append(condition.mode)
                foundMode = True
        if foundMode == False:
            unfoundDestModeTypes.append(modeType)

    unfoundModes = []
    for modeType in unfoundSourceModeTypes:
        unfoundModes.append(modes[modeType])
    unfoundModeCombinations = itertools.product(*unfoundModes)
    sourceVertices = []
    for unfoundModeCombo in unfoundModeCombinations:
        sourceVertex = sourceModes.copy()
        sourceVertex.extend(unfoundModeCombo)
        sourceVertices.append(sourceVertex)

    unfoundModes = []
    for modeType in unfoundDestModeTypes:
        unfoundModes.append(modes[modeType])
    unfoundModeCombinations = itertools.product(*unfoundModes)
    destVertices = []
    for unfoundModeCombo in unfoundModeCombinations:
        destVertex = destModes.copy()
        destVertex.extend(unfoundModeCombo)
        destVertices.append(destVertex)

    edges = []
    for source in sourceVertices:
        sourceindex = getIndex(source, vertices)
        for dest in destVertices:
            destindex = getIndex(dest, vertices)
            edges.append(Edge(sourceindex, destindex, guards, resets))
    
    return edges

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
    paths, vars, modes = walktree(code, tree)

    #print("Paths found:")
    #for result in paths:
    #    for item in result:
            #item.print()
            #print(item.mode)
            #print(item.modeType)
    #    print()

    print("Modes found: ")
    print(modes)

    output_dict.update(input_json)
    
    #TODO: create graph!
    vertices = []
    vertexStrings = []
    for vertex in itertools.product(*modes.values()):
        vertices.append(vertex)
        vertexstring = vertex[0]
        for index in range(1,len(vertex)):
            vertexstring += ";" + vertex[index]
        vertexStrings.append(vertexstring)

    edges = []
    guards = []
    resets = []

    for path in paths:
        transitions = createTransition(path, vertices, modes)
        for edge in transitions:
            edges.append([edge.source, edge.dest])
            guards.append(guardString(edge.guards))
            resets.append(resetString(edge.resets))
   
    output_dict['vertex'] = vertices
    #print(vertices)
    output_dict['variables'] = vars
    # #add edge, transition(guards) and resets
    output_dict['edge'] = edges
    #print(edges)
    output_dict['guards'] = guards
    #print(guards)
    output_dict['resets'] = resets
    print(resets)

    #output_json = json.dumps(output_dict, indent=4)
    #outfile = open(output_file_name, "w")
    #outfile.write(output_json)
    #outfile.close()

    print("wrote json to " + output_file_name)

