#Accidentally uploaded old code and left computer with newest code at home!!!! This is slightly old!

import clang.cindex
import typing
import json
import sys
#from cfg import CFG
#import clang.cfg


#index = clang.cindex.Index.create()
#translation_unit = index.parse('my_source.c', args=['-std=c99'])

#print all the cursor types
#for i in translation_unit.get_tokens(extent=translation_unit.cursor.extent):
#    print (i.kind)

#get the structs
def filter_node_list_by_node_kind(
    nodes: typing.Iterable[clang.cindex.Cursor],
    kinds: list
) -> typing.Iterable[clang.cindex.Cursor]:
    result = []
    for i in nodes:
        if i.kind in kinds:
            result.append(i)
    return result

#exclude includes
def filter_node_list_by_file(
    nodes: typing.Iterable[clang.cindex.Cursor],
    file_name: str
) -> typing.Iterable[clang.cindex.Cursor]:
    result = []
    for i in nodes:
        if i.location.file.name == file_name:
            result.append(i)
    return result


#all_classes = filter_node_list_by_node_kind(translation_unit.cursor.get_children(), [clang.cindex.CursorKind.ENUM_DECL, clang.cindex.CursorKind.STRUCT_DECL])
#for i in all_classes:
#    print (i.spelling)

#captured fields in classes
def is_exposed_field(node):return node.access_specifier == clang.cindex.AccessSpecifier.PUBLIC
def find_all_exposed_fields(
    cursor: clang.cindex.Cursor
):
    result = []
    field_declarations = filter_node_list_by_node_kind(cursor.get_children(), [clang.cindex.CursorKind.FIELD_DECL])
    for i in field_declarations:
        if not is_exposed_field(i):
            continue
        result.append(i.displayname)
    return result


def populate_field_list_recursively(class_name: str,class_field_map,class_inheritance_map):
    field_list = class_field_map.get(class_name)
    if field_list is None:
        return []
    baseClasses = class_inheritance_map[class_name]
    for i in baseClasses:
        field_list = populate_field_list_recursively(i.spelling) + field_list
    return field_list
#rtti_map = {}



def get_state_variables(translation_unit):
    rtti_map = {}
    source_nodes = filter_node_list_by_file(translation_unit.cursor.get_children(), translation_unit.spelling)
    all_classes = filter_node_list_by_node_kind(source_nodes, [clang.cindex.CursorKind.ENUM_DECL, clang.cindex.CursorKind.STRUCT_DECL])
    class_inheritance_map = {}
    class_field_map = {}
    for i in all_classes:
        bases = []
        for node in i.get_children():
            if node.kind == clang.cindex.CursorKind.CXX_BASE_SPECIFIER:
                referenceNode = node.referenced
                bases.append(node.referenced)
        class_inheritance_map[i.spelling] = bases
    for i in all_classes:
        fields = find_all_exposed_fields(i)
        class_field_map[i.spelling] = fields
    #print("foo")
    for class_name, class_list in class_inheritance_map.items():
        rtti_map[class_name] = populate_field_list_recursively(class_name,class_field_map,class_inheritance_map)
    for class_name, field_list in rtti_map.items():
        rendered_fields = []
        for f in field_list:
            rendered_fields.append(f)
    return rendered_fields



#def populate_field_list_recursively(class_name: str):
#    field_list = class_field_map.get(class_name)
#    if field_list is None:
#        return []
#    baseClasses = class_inheritance_map[class_name]
#    for i in baseClasses:
#        field_list = populate_field_list_recursively(i.spelling) + field_list
#    return field_list
#rtti_map = {}

#def get_state_variables():
#    print("foo")
#    for class_name, class_list in class_inheritance_map.items():
#        rtti_map[class_name] = populate_field_list_recursively(class_name)
#    for class_name, field_list in rtti_map.items():
#        rendered_fields = []
#        for f in field_list:
#            rendered_fields.append(f)
#    return rendered_fields



#how can we get the structure with the if statements
#need to get ahold of if statement, then using if statement methods, we can get the inner statement


#all_ifs = filter_node_list_by_node_kind(translation_unit.cursor.get_children(), [clang.cindex.CursorKind.IF_STMT])
#for i in all_ifs:
#    print (i.spelling)

#print(translation_unit.spelling)

def find_ifstmt(node):
    if node.kind.is_statement():
        #ref_node = clang.cindex.Cursor_ref(node)
        #print ("found %s"" % node.location.line)
        print(node.displayname)
        print(node.location.line)
    for c in node.get_children():
        find_ifstmt(c)

#finds the if statement, but cant get anything out of it
#find_ifstmt(translation_unit.cursor)


def rectree(node, indent):
    print("%s item %s of kind %s" % (indent, node.spelling, node.kind))
    for c in node.get_children():
        rectree(c, indent + "  ")
#goes through every part of code but doesn't give info about non variable names
#rectree(translation_unit.cursor, "")

#CursorKind.IF_STMT

#given the if statement as node, visit each of the else if statements
def visit_elif(node):
    print("visiting inner elseif statement")
    for i in node.get_children():
        #print(i.kind)
        #print(i.spelling)
        if i.kind == clang.cindex.CursorKind.IF_STMT or i.kind == clang.cindex.CursorKind.COMPOUND_STMT:
            print("found el-if of compound")
            print(i.spelling)
            visit_elif(i)
    print("finished elseif statement")    
#if (node.hasElseStorage()):
        #print("foo")


#TODO, but a method to get us the info from an if statemen
def visitif(node):
    print(node.spelling)

def visit_inner_if(node):
    for i in node.get_children():
        if i.kind == clang.cindex.CursorKind.IF_STMT:
            print("inner if %s" % i.kind)
            visit_elif(node)
        else:
            visit_inner_if(i)

def get_cursor_id(cursor, cursor_list = []):
    if cursor is None:
        return None

    # FIXME: This is really slow. It would be nice if the index API exposed
    # something that let us hash cursors.
    for i,c in enumerate(cursor_list):
        if cursor == c:
            return i
    cursor_list.append(cursor)
    return len(cursor_list) - 1

def get_info(node, depth=0):
    children = [get_info(c, depth+1)
                    for c in node.get_children()]
    return { 'id' : get_cursor_id(node),
             'kind' : node.kind,
             'usr' : node.get_usr(),
             'spelling' : node.spelling,
             'location' : node.location,
             'extent.start' : node.extent.start,
             'extent.end' : node.extent.end,
             'is_definition' : node.is_definition(),
             'definition id' : get_cursor_id(node.get_definition())}
             #'children' : children }

def code_from_cursor(cursor):
    code = []
    line = ""
    prev_token = None
    for tok in cursor.get_tokens():
        if prev_token is None:
            prev_token = tok
        prev_location = prev_token.location
        prev_token_end_col = prev_location.column + len(prev_token.spelling)
        cur_location = tok.location
        if cur_location.line > prev_location.line:
            code.append(line)
            line = " " * (cur_location.column - 1)
        else:
            if cur_location.column > (prev_token_end_col):
                line += " "
        line += tok.spelling
        prev_token = tok
    if len(line.strip()) > 0:
        code.append(line)
    return code


def visit_outter_if(node):
    if_statements = []
    for i in node.get_children():
        if i.kind == clang.cindex.CursorKind.IF_STMT:
            #print("Outter if statement %s" % i.kind)
            #visit_inner_if(i)
            if_statements.append(code_from_cursor(i)[0])
            #print(if_statements)
            #print(i.spelling)
            #for child in i.get_children():
            #    print(str(i.kind) + str(i.spelling))
        else:
            out_ifs = visit_outter_if(i)
            for statement in out_ifs:
                if_statements.append(statement)
    return if_statements

#visit_outter_if(translation_unit.cursor)


def get_outter_if_cursors(node):
    if_statements = []
    for i in node.get_children():
        if i.kind == clang.cindex.CursorKind.IF_STMT:
            #print("Outter if statement %s" % i.kind)
            #visit_inner_if(i)
            if_statements.append(i)
            #print(if_statements)
            #print(i.spelling)
            #for child in i.get_children():
            #    print(str(i.kind) + str(i.spelling))
        else:
            out_ifs = get_outter_if_cursors(i)
            for statement in out_ifs:
                if_statements.append(statement)
    return if_statements


def get_inner_if_cursors(node):
    if_statements = []
    for i in node.get_children():
        #print(i.kind)
        if i.kind == clang.cindex.CursorKind.IF_STMT:
            if_statements.append(i)
            #print(if_statements)
            #print(i.spelling)
            #for child in i.get_children():
            #    print(str(i.kind) + str(i.spelling))
        else:
            out_ifs = get_inner_if_cursors(i)
            for statement in out_ifs:
                if_statements.append(statement)
    return if_statements



#input: current node, the set of guards/conditionals up to this point, the set of resets
#return: the sets of guards/resets for all the children
def traverse_tree(node, guards, resets, indent,hasIfParent):
    #print(guards)
    childrens_guards=[]
    childrens_resets=[]
    results = []
    found = []
    #print(resets)
    #if (indent== '------'):
    #    print(guards)
    #    return [guards,resets]
    #if (node.kind == clang.cindex.CursorKind.RETURN_STMT):
    #    return [guards, resets]
    for i in node.get_children():
        if i.kind == clang.cindex.CursorKind.IF_STMT: #TODO: add else statements
            temp_results = traverse_tree(i, guards, resets, indent+ "-", True)
            for item in temp_results:
                found.append(item)
            #childrens_guards.append(result_guards)
            #childrens_resets.append(result_resets)

        else:
            reset_copy = resets
            guard_copy = guards
            if hasIfParent:
                if node.kind == clang.cindex.CursorKind.IF_STMT and i.kind != clang.cindex.CursorKind.COMPOUND_STMT:
                    childrens_guards.append(code_from_cursor(i))
                    print(indent + "Guard statement")
                    print(code_from_cursor(i))
                    #found.append(code_from_cursor(i))
                    #temp_results = traverse_tree(i, guard_copy, reset_copy, indent+ "-", hasIfParent)
                    #for result in temp_results:
                    #    result.append(code_from_cursor(i))
                    #    results.append(result)
                    #if len(temp_results)== 0:
                    #    result.append([code_from_cursor(i)])
                    #    print("foo")
                if node.kind == clang.cindex.CursorKind.COMPOUND_STMT and i.kind != clang.cindex.CursorKind.DECL_STMT:
                    print(indent + "Reset statement")
                    print(code_from_cursor(i))
                    childrens_resets.append(code_from_cursor(i))
                    #found.append(code_from_cursor(i))
                    #temp_results = traverse_tree(i, guard_copy, reset_copy, indent+ "-", hasIfParent)
                    #for result in temp_results:
                    #    result.append(code_from_cursor(i))
                    #    results.append(result)
                    #if len(temp_results) == 0:
                    #    results.append([code_from_cursor(i)])
                    #    print("foo")
                #else:
            temp_results = traverse_tree(i, guard_copy, reset_copy, indent+ "-", hasIfParent)
            for item in temp_results:
                 found.append(item)
            #childrens_guards.append(results[0])
            #childrens_resets.append(results[1])
    #print(results)
    #output = []
    #if len(results) < 0:
    #    input = []
    #    input.append(childrens_results)
    #    input.append(childrens_guards)
    #    output.append(input)
    #else:
    #    for result in results:
    #        result.append(childrens_resets)
    #        result.append(childrens_guards)
    #        output.append(result)
    #print(output)
    #ASSUMPTION: lowest level is a reset!!!
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

#assumption, word mode is not in states
def pretty_vertices(vertices):
    output = []
    for vertex_code in vertices:
        parts = vertex_code.split("==")
        nonwhitespace = parts[-1].split()
        if "mode" not in nonwhitespace:
            output.append(nonwhitespace[0].strip(')'))
    return output
 
def get_next_state(code):
    line = code[-2]
    parts = line.split("=")
    state = parts[-1].strip(';')
    return state.strip()


#TODO: consider or??
def pretty_guards(code):
    line = code[0]
    conditional = line.strip('if').strip('{')
    conditions = conditional.split('&&')
    output = "And"
    for condition in conditions: 
        output += condition.strip() + ","
    output = output.strip(",")
    return output

#assumption: last two lines of code are reset and bracket... not idea
def pretty_resets(code):
    outstring = ""
    for index in range(0,len(code)):
       if index != 0 and index != len(code)-1 and index != len(code)-2:
        outstring += code[index].strip().strip('\n')
    return outstring.strip(';')


##main code###
#print(sys.argv)
if len(sys.argv) < 4:
    print("incorrect usage. call createGraph.py program inputfile outputfilename")
    quit()

input_code_name = sys.argv[1]
input_file_name = sys.argv[2] 
output_file_name = sys.argv[3]



with open(input_file_name) as in_json_file:
    input_json = json.load(in_json_file)

output_dict = {
}

output_dict.update(input_json)

#file parsing set up
index = clang.cindex.Index.create()
translation_unit = index.parse(input_code_name, args=['-std=c99'])
print("testing cfg")

#FunctionDecl* func = /* ... */ ;
#ASTContext* context = /* ... */ ;
#Stmt* funcBody = func->getBody();
#CFG* sourceCFG = CFG::buildCFG(func, funcBody, context, clang::CFG::BuildOptions());

#add each variable in state struct as variable
variables = get_state_variables(translation_unit)
output_dict['variables'] = variables

#traverse the tree for all the paths
paths = traverse_tree(translation_unit.cursor, [],[], "", False)

vertices = []
counter = 0
for path in paths:
    vertices.append(str(counter))
    counter += 1

output_dict['vertex'] = vertices


#traverse outter if statements and find inner statements
edges = []
guards = []
resets = []

#add edge, transition(guards) and resets
output_dict['edge'] = edges
output_dict['guards'] = guards
output_dict['resets'] = resets

output_json = json.dumps(output_dict, indent=4)
#print(output_json)
outfile = open(output_file_name, "w")
outfile.write(output_json)
outfile.close()

print("wrote json to " + output_file_name)










