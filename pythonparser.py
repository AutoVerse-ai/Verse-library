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
    '''
    Initalizing function for a controller_ast object.
    Reads in the code and parses it to a python ast and statement tree.
    Statement tree is a tree of nodes that contain a list in their data. The list contains a single guard or a list of resets.
    Variables (inputs to the controller) are collected.
    Modes are collected from all enums that have the word "mode" in them.
    Vertices are generated by taking the products of mode types. 
    '''
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(code)
        self.statementtree, self.variables, self.modes = self.initalwalktree(code, self.tree)
        self.vertices = []
        self.vertexStrings = []
        for vertex in itertools.product(*self.modes.values()):
            self.vertices.append(vertex)
            vertexstring = vertex[0]
            for index in range(1,len(vertex)):
                vertexstring += ";" + vertex[index]
            self.vertexStrings.append(vertexstring)
        self.paths = None


    '''
    Function to populate paths variable with all paths of the controller.
    '''
    def getAllPaths(self):
        currentModes = []
        for modeTypes in self.modes.values():
            currentModes.extend(modeTypes)
        self.paths = self.getNextModes(currentModes)
        return self.paths
    
    '''
    getNextModes takes in a list of current modes. It should include all modes. 
    getNextModes returns a list of paths that can be followed when in the given mode.
    A path is a list of statements, all guards and resets along the path. They are in the order they are encountered in the code.
    TODO: should we not force all modes be listed? Or rerun for each unknown/don't care node? Or add them all to the list
    '''
    def getNextModes(self, currentModes):
        #walk the tree and capture all paths that have modes that are listed. Path is a list of statements
        paths = []
        rootid = self.statementtree.root
        currnode = self.statementtree.get_node(rootid)
        paths = self.walkstatements(currnode, currentModes)
        
        return paths 

    '''
    Helper function to walk the statement tree from parentnode and find paths that are allowed in the currentMode.
    Returns a list of paths. 
    '''
    def walkstatements(self, parentnode, currentModes):
        nextsPaths = []

        for node in self.statementtree.children(parentnode.identifier):
            statement = node.data
            
            if isinstance(statement[0], Guard) and statement[0].isModeCheck():
                    if statement[0].mode in currentModes:
                        #print(statement.mode)
                        newPaths = self.walkstatements(node, currentModes)
                        for path in newPaths:
                            newpath = statement.copy()
                            newpath.extend(path)
                            nextsPaths.append(newpath)
                        if len(nextsPaths) == 0:
                            nextsPaths.append(statement)
        
            else:
                newPaths =self.walkstatements(node, currentModes)
                for path in newPaths:
                    newpath = statement.copy()
                    newpath.extend(path)
                    nextsPaths.append(newpath)
                if len(nextsPaths) == 0:
                            nextsPaths.append(statement)
 
        return nextsPaths


    '''
    Function to create a json of the full graph.
    Requires that paths class variables has been set. 
    '''
    def create_json(self, input_file_name, output_file_name):
        if not self.paths:
            print("Cannot call create_json without calling getAllPaths")
            return

        with open(input_file_name) as in_json_file:
            input_json = json.load(in_json_file)

        output_dict = {
        }

        output_dict.update(input_json)

        edges = []
        guards = []
        resets = []

        for path in self.paths:
            transitions = createTransition(path, self.vertices, self.modes)
            for edge in transitions:
                edges.append([edge.source, edge.dest])
                guards.append(guardString(edge.guards))
                resets.append(resetString(edge.resets))
    
        output_dict['vertex'] = self.vertexStrings
        #print(vertices)
        output_dict['variables'] = self.variables
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
    '''
    Function called by init function. Walks python ast and parses to a statement tree. 
    Returns a statement tree (nodes contain a list of either a single guard or muliple resets), the variables, and a mode dictionary
    '''
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
                    statementtree = self.parsenodelist(code, node.body, False, Tree(), None)
                    #print(type(node.args))
                    args = node.args.args
                    for arg in args:
                        if "mode" not in arg.arg:
                            vars.append(arg.arg)
                            #todo: what to add for return values
        return [statementtree, vars, mode_dict]


    '''
    Helper function for initalwalktree which parses the statements in the controller function into a statement tree
    '''
    def parsenodelist(self, code, nodes, addResets, tree, parent):
        childrens_guards=[]
        childrens_resets=[]
        recoutput = []
        #tree.show()
        if parent == None:
            s = Statement("root", None, None)
            tree.create_node("root")
            parent = tree.root

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
                newTree.create_node(tag= guard.code, data = [guard])
                #print(self.nodect)
                tempresults = self.parsenodelist(code, childnode.body, True, newTree, newTree.root)
                #for result in tempresults:
                recoutput.append(tempresults)

        
        #pathsafterme = [] 
        if len(childrens_resets) > 0:
            #print("adding node:" + str(self.nodect) + "with parent:" + str(parent))
            tree.create_node(tag = childrens_resets[0].code, data = childrens_resets, parent= parent)
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

    #read in the controler code
    f = open(input_code_name,'r')
    code = f.read()

    #parse the controller code into our controller ast objct
    controller_obj = controller_ast(code)

    #demonstrate you can check getNextModes after only initalizing
    paths = controller_obj.getNextModes("NormalA;Normal3")
   
    print("Results")
    for path in paths:
        for item in path:
            print(item.code)
        print()
    print("Done")

    #attempt to write to json, fail because we haven't populated paths yet
    controller_obj.create_json(input_file_name, output_file_name)

    controller_obj.getAllPaths()

    controller_obj.create_json(input_file_name, output_file_name)


    
    

