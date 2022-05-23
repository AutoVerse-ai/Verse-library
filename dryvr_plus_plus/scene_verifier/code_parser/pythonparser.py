#parse python file

#REQUIRES PYTHON 3.8!
from cgitb import reset
#import clang.cindex
import typing
import json
import sys
from typing import List, Tuple, Dict
import re
import itertools
import ast

from treelib import Node, Tree

class SimVarSet:
    """Variable/member set needed for simulation/verification for some object"""
    cont: List[str]     # Continuous variables
    disc: List[str]     # Discrete variables
    def __init__(self):
        self.cont = []
        self.disc = []

    def add_cont(self, s: str):
        self.cont.append(s)

    def add_disc(self, s: str):
        self.disc.append(s)

class ParseVarSet(SimVarSet):
    """Variable/member set encountered during parsing for some object"""
    static: List[str]   # Static data in object
    def __init__(self):
        super().__init__()
        self.static = []

    def add_static(self, s: str):
        self.static.append(s)

'''
Edge class: utility class to hold the source, dest, guards, and resets for a transition
'''
class Edge:
    def __init__(self, source, dest, guards, resets):
        self.source = source
        self.dest = dest
        self.guards = guards
        self.resets = resets

'''
Statement super class. Holds the code and mode information for a statement.
If there is no mode information, mode and modeType are None.
'''
class Statement:
    def __init__(self, code, mode, modeType, func = None, args = None):
        self.code = code
        self.modeType = modeType
        self.mode = mode
        self.func = func
        self.args = args

    def print(self):
        print(self.code)


'''
Guard class. Subclass of statement.
'''
class Guard(Statement):
    def __init__(self, code, mode, modeType, inp_ast, func=None, args=None):
        super().__init__(code, mode, modeType, func, args)
        self.ast = inp_ast


    '''
    Returns true if a guard is checking that we are in a mode.
    '''
    def isModeCheck(self):
        return self.modeType != None

    '''
    Helper function to parse a node that contains a guard. Parses out the code and mode.
    Returns a Guard.
    TODO: needs to handle more complex guards.
    '''
    def parseGuard(node, code):
        #assume guard is a strict comparision (modeType == mode)
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.comparators[0], ast.Attribute):
                if ("Mode" in str(node.test.comparators[0].value.id)):
                    modeType = str(node.test.comparators[0].value.id)
                    mode = str(node.test.comparators[0].attr)
                    return Guard(ast.get_source_segment(code, node.test), mode, modeType, node.test)
            else:
                return Guard(ast.get_source_segment(code, node.test), None, None, node.test)
        elif isinstance(node.test, ast.BoolOp):
            return Guard(ast.get_source_segment(code, node.test), None, None, node.test)
        elif isinstance(node.test, ast.Call):
            source_segment = ast.get_source_segment(code, node.test)
            if "map" in source_segment:
                func = node.test.func.value.id + '.' + node.test.func.attr
                args = []
                for arg in node.test.args:
                    args.append(arg.value.id + '.' + arg.attr)
                return Guard(source_segment, None, None, node.test, func, args)

'''
Reset class. Subclass of statement.
'''
class Reset(Statement):
    def __init__(self, code, mode, modeType, inp_ast):
        super().__init__(code, mode, modeType)
        self.ast = inp_ast

    '''
    Returns true if a reset is updating our mode.
    '''
    def isModeUpdate(self):
        return self.modeType != None

    '''
    Helper function to parse a node that contains a reset. Parses out the code and mode.
    Returns a reset.
    '''
    def parseReset(node, code):
        #assume reset is modeType = newMode
        if isinstance(node.value, ast.Attribute):
            #print("resets " + str(node.value.value.id))
            #print("resets " + str(node.value.attr))
            if ("Mode" in str(node.value.value.id)):
                modeType = str(node.value.value.id)
                mode = str(node.value.attr)
            return Reset(ast.get_source_segment(code, node), mode, modeType, node)
        return Reset(ast.get_source_segment(code, node), None, None, node)


'''
Util class to handle building transitions given a path.
'''
class TransitionUtil:
    '''
    Takes in a list of reset objects. Returns a string in the format json expected.
    '''
    def resetString(resets):
        outstr = ""
        for reset in resets:
            outstr+= reset.code + ";"
        outstr = outstr.strip(";")
        return outstr

    '''
    Takes in guard code. Returns a string in the format json expected.
    TODO: needs to handle more complex guards.
    '''
    def parseGuardCode(code):
        parts = code.split("and")
        out = code
        if len(parts) > 1:
            left = TransitionUtil.parseGuardCode(parts[0])
            right = TransitionUtil.parseGuardCode(parts[1])
            out = "And(" + left + "," + right + ")"
        return out

    '''
    Helper function for parseGuardCode.
    '''
    def guardString(guards):
        output = ""
        first = True
        for guard in guards:
            #print(type(condition))
            if first:
                output+= TransitionUtil.parseGuardCode(guard.code)
            else:
                output = "And(" + TransitionUtil.parseGuardCode(guard.code) + ",(" + output + "))"
            first = False
        return output


    '''
    Helper function to get the index of the vertex for a set of modes.
    Modes is a list of all modes in the current vertex.
    Vertices is the list of vertices.
    TODO: needs to be tested on more complex examples to see if ordering stays and we can use index function
    '''
    def getIndex(modes, vertices):
        return vertices.index(tuple(modes))

    '''
    Function that creates transitions given a path.
    Will create multiple transitions if not all modeTypes are checked/set in the path.
    Returns a list of edges that correspond to the path.
    '''
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
            sourceindex = TransitionUtil.getIndex(source, vertices)
            for dest in destVertices:
                destindex = TransitionUtil.getIndex(dest, vertices)
                edges.append(Edge(sourceindex, destindex, guards, resets))
        return edges

class ControllerAst():
    '''
    Initalizing function for a controllerAst object.
    Reads in the code and parses it to a python ast and statement tree.
    Statement tree is a tree of nodes that contain a list in their data. The list contains a single guard or a list of resets.
    Variables (inputs to the controller) are collected.
    Modes are collected from all enums that have the word "mode" in them.
    Vertices are generated by taking the products of mode types.
    '''
    vars_dict: Dict[str, ParseVarSet]
    modes: Dict[str, List[str]]
    def __init__(self, code = None, file_name = None):
        assert code is not None or file_name is not None
        if file_name is not None:
            with open(file_name,'r') as f:
                code = f.read()

        self.code = code
        self.tree = ast.parse(code)
        self.statementtree, self.modes, self.vars_dict = self.initalwalktree(code, self.tree)
        self.vertices = []
        self.vertexStrings = []
        for vertex in itertools.product(*self.modes.values()):
            self.vertices.append(vertex)
            vertexstring = ','.join(vertex)
            self.vertexStrings.append(vertexstring)
        self.paths = None

    '''
    Function to populate paths variable with all paths of the controller.
    '''
    def getAllPaths(self):
        self.paths = self.getNextModes([], True)
        return self.paths

    '''
    getNextModes takes in a list of current modes. It should include all modes.
    getNextModes returns a list of paths that can be followed when in the given mode.
    A path is a list of statements, all guards and resets along the path. They are in the order they are encountered in the code.
    TODO: should we not force all modes be listed? Or rerun for each unknown/don't care node? Or add them all to the list
    '''
    def getNextModes(self, currentModes: List[str], getAllPaths= False) -> List[str]:
        #walk the tree and capture all paths that have modes that are listed. Path is a list of statements
        paths = []
        rootid = self.statementtree.root
        currnode = self.statementtree.get_node(rootid)
        paths = self.walkstatements(currnode, currentModes, getAllPaths)

        return paths

    '''
    Helper function to walk the statement tree from parentnode and find paths that are allowed in the currentMode.
    Returns a list of paths.
    '''
    def walkstatements(self, parentnode, currentModes, getAllPaths):
        nextsPaths = []

        for node in self.statementtree.children(parentnode.identifier):
            statement = node.data

            if isinstance(statement[0], Guard) and statement[0].isModeCheck():
                if getAllPaths or statement[0].mode in currentModes:
                    #print(statement.mode)
                    newPaths = self.walkstatements(node, currentModes, getAllPaths)
                    for path in newPaths:
                        newpath = statement.copy()
                        newpath.extend(path)
                        nextsPaths.append(newpath)
                    if len(nextsPaths) == 0:
                        nextsPaths.append(statement)

            else:
                newPaths =self.walkstatements(node, currentModes, getAllPaths)
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
            transitions = TransitionUtil.createTransition(path, self.vertices, self.modes)
            for edge in transitions:
                edges.append([edge.source, edge.dest])
                guards.append(TransitionUtil.guardString(edge.guards))
                resets.append(TransitionUtil.resetString(edge.resets))

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
        mode_dict: Dict[str, List[str]] = {}
        state_object_dict: Dict[str, ParseVarSet] = {}
        vars_dict: Dict[str, ParseVarSet] = {}
        statementtree = Tree()
        for node in ast.walk(tree): #don't think we want to walk the whole thing because lose ordering/depth
            # Get all the modes
            if isinstance(node, ast.ClassDef):
                if "Mode" in node.name:
                    mode_dict[node.name] = [str(modeValue.targets[0].id) for modeValue in node.body]
            if isinstance(node, ast.ClassDef):
                if "State" in node.name:
                    state_object_dict[node.name] = ParseVarSet()
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if "init" in item.name:
                                for arg in item.args.args:
                                    var_name = arg.arg
                                    if "self" not in var_name:
                                        if "type" == var_name:
                                            state_object_dict[node.name].add_static(var_name)
                                        elif "mode" not in var_name:
                                            state_object_dict[node.name].add_cont(var_name)
                                        else:
                                            state_object_dict[node.name].add_disc(var_name)
            if isinstance(node, ast.FunctionDef):
                if node.name == 'controller':
                    #print(node.body)
                    statementtree = self.parsenodelist(code, node.body, False, Tree(), None)
                    #print(type(node.args))
                    args = node.args.args
                    for arg in args:
                        if arg.annotation is None:
                            continue
                        if arg.annotation.id not in state_object_dict:
                            continue
                        arg_annotation = arg.annotation.id
                        arg_name = arg.arg
                        vars_dict[arg_name] = state_object_dict[arg_annotation]
        return [statementtree, mode_dict, vars_dict]


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

class EmptyAst(ControllerAst):
    def __init__(self):
        super().__init__(code="True", file_name=None)
        self.modes = {
            'NullMode':['Null'],
            'LaneMode':['Normal']
        }
        self.paths = None
        self.vars_dict = []
        self.vertexStrings = ['Null,Normal']
        self.vertices=[('Null','Normal')]
        self.statementtree.create_node("root")

##main code###
if __name__ == "__main__":
    #if len(sys.argv) < 4:
    #    print("incorrect usage. call createGraph.py program inputfile outputfilename")
    #    quit()

    input_code_name = sys.argv[1]
    input_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    with open(input_file_name) as in_json_file:
        input_json = json.load(in_json_file)

    output_dict = {
    }

    #read in the controler code
    f = open(input_code_name,'r')
    code = f.read()

    #parse the controller code into our controller ast objct
    controller_obj = ControllerAst(code)

    print(controller_obj.variables)

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

    #call function that gets all paths
    controller_obj.getAllPaths()

    #write json with all paths
    controller_obj.create_json(input_file_name, output_file_name)





