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

from treelib import Node, Tree

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
                    if "mode" not in arg.arg:
                        vars.append(arg.arg)
                        #todo: what to add for return values
    return [out, vars, mode_dict]



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

def parseGuardCode(code):
    #TODO: should be more general and handle or
    parts = code.split("and")
    out = code
    if len(parts) > 1:
        left = parseGuardCode(parts[0])
        right = parseGuardCode(parts[1])
        out = "And(" + left + "," + right + ")"
    return out

def guardString(guards):
    output = ""
    first = True
    for guard in guards: 
        #print(type(condition))
        if first:
            output+= parseGuardCode(guard.code)
        else:
            output = "And(" + parseGuardCode(guard.code) + ",(" + output + "))"
        first = False
    return output


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

class controller_ast():
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(code)
        self.nodect= 1
        self.statementtree, self.variables, self.modes = self.initalwalktree(code, tree)
        self.vertices = []
       
    
    #assume that currentModes is a list of all node types!
    #TODO: should we not force all modes be listed? Or rerun for each unknown/don't care node? Or add them all to the list
    def getNextModes(self, currentModes):
        #walk the tree and capture all paths that have modes that are listed. Path is a list of statements
        paths = []
        rootid = self.statementtree.root
        currnode = self.statementtree.get_node(rootid)
        paths = self.walkstatements(currnode, currentModes)
        
        return paths 

    def walkstatements(self, parentnode, currentModes):
        nextsPaths = []
        for node in self.statementtree.children(parentnode.identifier):
            statement = node.tag
            print(statement)
            print(node)
            if isinstance(statement, Guard) and statement.isModeCheck():
                    if statement.mode in currentModes:
                        newPaths = self.walkstatements(node, currentModes)
                        for path in newPaths:
                            nextsPaths.append(statement.extend(path))
                        if len(nextsPaths) == 0:
                            nextsPaths.append(statement)
        
            else:
                newPaths =self.walkstatements(node, currentModes)
                for path in newPaths:
                    nextsPaths.append(statement.extend(path))
                if len(nextsPaths) == 0:
                            nextsPaths.append(statement)
        
        return nextsPaths


    def create_json(input_file_name, output_file_name):
        with open(input_file_name) as in_json_file:
            input_json = json.load(in_json_file)

        output_dict = {
        }

        output_dict.update(input_json)

        edges = []
        guards = []
        resets = []

        for path in paths:
            transitions = createTransition(path, vertices, modes)
            for edge in transitions:
                edges.append([edge.source, edge.dest])
                guards.append(guardString(edge.guards))
                resets.append(resetString(edge.resets))
    
        output_dict['vertex'] = vertexStrings
        #print(vertices)
        output_dict['variables'] = variables
        # #add edge, transition(guards) and resets
        output_dict['edge'] = edges
        #print(len(edges))
        output_dict['guards'] = guards
        #print(len(guards))
        output_dict['resets'] = resets
        #print(len(resets))

        output_json = json.dumps(output_dict, indent=4)
        outfile = open(output_file_name, "w")
        outfile.write(output_json)
        outfile.close()

        print("wrote json to " + output_file_name)

    #inital tree walk, parse into a tree of resets/modes
    def initalwalktree(self, code, tree):
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
                    out = self.parsenodelist(code, node.body, False, Tree(), None)
                    #print(type(node.args))
                    args = node.args.args
                    for arg in args:
                        if "mode" not in arg.arg:
                            vars.append(arg.arg)
                            #todo: what to add for return values
        return [out, vars, mode_dict]



    def parsenodelist(self, code, nodes, addResets, tree, parent):
        childrens_guards=[]
        childrens_resets=[]
        recoutput = []
        #tree.show()
        if parent == None:
            s = Statement("root", None, None)
            tree.create_node(s, self.nodect)
            parent = self.nodect
            self.nodect += 1

        for childnode in nodes:
            if isinstance(childnode, ast.Assign) and addResets:
                reset = Reset.parseReset(childnode, code)
                #print("found reset: " + reset.code)
                childrens_resets.append(reset)
            if isinstance(childnode, ast.If):
                guard = Guard.parseGuard(childnode, code)
                childrens_guards.append(guard)
                #print("found if statement: " + guard.code)
                newTree = Tree()
                temp_node_num = self.nodect
                self.nodect += 1
                newTree.create_node([guard], temp_node_num)
                #print(self.nodect)
                tempresults = self.parsenodelist(code, childnode.body, True, newTree, temp_node_num)
                #for result in tempresults:
                recoutput.append(tempresults)

        
        #pathsafterme = [] 
        if len(childrens_resets) > 0:
            #print("adding node:" + str(self.nodect) + "with parent:" + str(parent))
            tree.create_node(childrens_resets, self.nodect, parent= parent)
            parent = self.nodect
            self.nodect += 1
        for subtree in recoutput:
            #print("adding subtree:" + " to parent:" + str(parent))
            tree.paste(parent, subtree)
                
        
        return tree


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
    #paths, variables, modes = walktree(code, tree)

    test = controller_ast(code)
    paths = test.getNextModes("NormalA;Normal3")
    variables = test.variables
    modes = test.modes

    for path in paths:
        for item in path:
            print(item.code)
        print()
    
    #print("Paths found:")
    #for result in paths:
    #    for item in result:
            #item.print()
            #print(item.mode)
            #print(item.modeType)
    #    print()

    #print("Modes found: ")
    #print(modes)

    output_dict.update(input_json)
    
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
   
    output_dict['vertex'] = vertexStrings
    #print(vertices)
    output_dict['variables'] = variables
    # #add edge, transition(guards) and resets
    output_dict['edge'] = edges
    #print(len(edges))
    output_dict['guards'] = guards
    #print(len(guards))
    output_dict['resets'] = resets
    #print(len(resets))

    output_json = json.dumps(output_dict, indent=4)
    outfile = open(output_file_name, "w")
    outfile.write(output_json)
    outfile.close()

    print("wrote json to " + output_file_name)

