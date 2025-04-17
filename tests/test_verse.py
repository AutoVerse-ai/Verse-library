# Introducing unittests for Verse development process
# Read more from https://docs.python.org/3/library/unittest.html
# Call the demo scenario
# A scenario is created for testing
import unittest
from ball_bounce_test import ball_bounce_test
from highway_test import highway_test
from verse import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNodeType
from enum import Enum, auto

import os
import subprocess
import importlib.util

'''
Goal: Using demo file (Verse_library/demo) as a test files. 
Observation: 
- Verse_library/demo contains many different models, in different folders
- In each of these folders, there is a file for simulate the scenarios
Problem
- The Verse team may add more folders or change current folders for other different scenarios in the future. 
Questions:

'''

def is_python_file(filepath):
    '''
    Helper function. Check if the filepath is pointing to a Python file
    Input: The filepath.

    Output: Whether this file is a Python file
    '''
    _, extension = os.path.splitext(filepath)
    python_extensions = ('.py')
    return extension.lower() in python_extensions

def get_python_file(filepath):
    '''
    Given a filepath, make sure this filepath is a folder. From this folder, return 
    the path to the python files inside this folder.

    Input: The filepath, path to the folder.

    Output: A list of Python file (.py) inside this folder
    '''
    if not os.path.exists(filepath):
        print(f"Error: Directory '{filepath}' does not exist.")
        return []
    if not os.path.isdir(filepath):
        print(f"Error: '{filepath}' is not a directory.")
        return []
    try:
        control_files = []
        entries = os.listdir(filepath)
        for entry in entries:
            if os.path.isdir(os.path.join(filepath, entry)):
                new_filepath = filepath + '/' + entry
                control_files = control_files + get_python_file(new_filepath)
            elif is_python_file(entry):
                control_files.append(filepath + '/' + entry)
        return control_files
    except Exception as e:
        print("Exceptions:", e)
        return []

def check_all_using_dependencies(python_filepath):
    '''
    Given a filepath of a Python program, read the file and 
    return all the dependencies that this Python program depends on

    Input: a filepath of a Python program

    Output: a list of dependencies that this program needs for running
    '''
    dependencies = []
    with open(python_filepath, "r") as file:
        for line in file:
            words = line.split()
            if len(words) > 0:
                if words[0] in ['import']:
                    dependencies.append(words[1])
                if words[0] in ['from']:
                    base = words[1].lower()
                    for i in range(3, len(words), 1):
                        if words[i] == '*':
                            dependencies.append(base)
                        else:
                            if words[i][len(words[i]) - 1] == ',':
                                dependencies.append(base + '.' + words[i][:-1].lower())
                            else:
                                dependencies.append(base + '.' + words[i].lower())
    return dependencies

def is_scenario_file(python_filepath):
    '''
    Given a filepath to a Python program, check if this program is 
    likely to be a file for scenario or simulation. 

    NOTE: In every file for scenario, there is a package which is always being imported.
    It's verse.scenario. So if verse.scenario is imported, the program is likely to be 
    made for scenario or simulation. 
    
    Use verse.scenario as a the keyword, if parts of verse.scenario are imported, then return True, else False

    Input: A filepath of a Python program

    Output: Whether the program is likely to be a file for scenario or simulation.
    '''
    dependencies = check_all_using_dependencies(python_filepath)
    keyword = 'verse.scenario'
    for dependency in dependencies:
        if dependency[:len(keyword)] == keyword:
            return True
    return False

def get_scenario_files_from_folder(filepath):
    '''
    Given a filepath to a folder, get all the scenarios files inside that folder.

    Input: A filepath to folder

    Output: A list of filepaths to scenario files inside that folder.
    '''
    scenario_files = []
    all_python_filepaths = get_python_file(filepath)
    for python_filepath in all_python_filepaths:
        if is_scenario_file(python_filepath):
            scenario_files.append(python_filepath)
    return scenario_files

def test_scenario(scenario_filepath):
    '''
    Given a filepath to a scenario file, check if it's able to run correctly. 

    Input: A filepath to a scenario file

    Output: All compilation errors that may occur when running the scenario.

    NOTE: Although I could also use os.system(), it seems subprocess allows me to
    show the errors and the output.
    '''
    errors = []
    print(f"Running '{scenario_filepath}'")
    result = subprocess.run(
                    ['python', scenario_filepath],
                    check=False,        
                    text=True,
                    capture_output=True
    )
    if result.stderr:
        print(f"There are errors '{scenario_filepath}'. Checks the summary.")
        errors.append((scenario_filepath, result.stderr))
    else:
        print(f"Compile '{scenario_filepath}' successfully")
        print("Output:")
        print(result.stdout)
    return errors

def run_test(dir):
    """
    Run unittest test files individually from the specified directory path.
    
    Args:
        filepath (str): Directory path containing unittest test files
    
    Returns:
        str: "SUCCESS" if all tests pass, "FAILURE" if any test fails
    """
    if not os.path.isdir(dir):
        raise ValueError(f"Directory '{dir}' does not exist")
    
    # Results dictionary to track success/failure
    results = {}
    
    # Print header
    print(f"Running unittest tests in directory: {dir}")
    print("-" * 50)
    
    # Get all test files
    test_files = [f for f in os.listdir(dir) if  f.endswith('.py')]
    
    if not test_files:
        print("No unittest test files found in directory", dir)
        return "SUCCESS"
    
    # Sort files for consistent execution order
    test_files.sort()
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Process each file individually
    for file_name in test_files:
        print(f"\nProcessing {file_name}")
        print("=" * 30)
        
        try:
            # Construct full file path
            file_path = os.path.join(dir, file_name)
            
            # Create module name from file name
            module_name = file_name[:-3]  # Remove .py extension
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Create a test suite for this file
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            # Run the tests for this file
            result = runner.run(suite)
            
            # Store result
            results[file_name] = result.wasSuccessful()
            
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            results[file_name] = False
    # Print results for debugging
    print("\nTest Results:")
    print(results)
    
    # Check if any test failed
    for file_name, success in results.items():
        if not success:
            return "FAILURE"
    return "SUCCESS"
class TestSimulatorMethods(unittest.TestCase):
    def setUp(self):
        pass

    # def test_m2_2c5n(self):
    #     trace = m2_2c5n_test()
    #     root = trace.root
    #     '''
    #     Test the max height
    #     '''
    #     max_height = 33
    #     assert trace.height(root) <= max_height
    #     print("Max height test passed!")
    #
    #
    #     '''
    #     Test the number of nodes
    #     '''
    #     assert len(trace.nodes) == 33
    #     print("Nodes number test passed!")
    # def testBallBounce(self):
    #     '''
    #     Test basic ball bounce scenario
    #     Test plotter
    #     '''
    #     trace, _ = ball_bounce_test()
    #     '''
    #     Test properties of root node
    #     '''
    #     root = trace.root
    #     baseAgent = BaseAgent.__init__()
    #     assert root.agent == baseAgent
    #     assert root.mode == ""
    #     assert root.start_time == 0
    #     print("Root test passed!")
    #
    #     '''
    #     Test the max height
    #     '''
    #     max_height = 15
    #     assert trace.height(root) <= max_height
    #     print("Max height test passed!")
    #
    #     '''
    #     Test properties of leaf node
    #     '''
    #     # leafs = self.get_leaf_nodes(root)
    #     # for leave in leafs:
    #     #     assert leave.agent == baseAgent
    #     #     assert leave.mode == ""
    #     #     assert leave.start_time == 0
    #     # print("Leave node test passed!")
    #
    #     '''
    #     Test the number of nodes
    #     '''
    #     #assert len(trace.nodes) == 10
    #     #print("Nodes number test passed!")

    # def testHighWay(self):
    #     '''
    #     Test highway scenario
    #     Test both simulation and verification function
    #     '''
    #     trace_sim, trace_veri = highway_test()

    #     # assert trace_sim.type == AnalysisTreeNodeType.SIM_TRACE
    #     # assert trace_veri.type == AnalysisTreeNodeType.REACH_TUBE

    # def test_demo_compiling(self):
    #     scenario_files = get_scenario_files_from_folder('demo/fp_demos')
    #     print(scenario_files)
    #     all_errors = []
    #     for scenario_file in scenario_files:
    #         errors = test_scenario(scenario_file)
    #         all_errors.extend(errors)
    #     if all_errors:
    #         print("\nSummary of Errors:")
    #         for filepath, error_msg in all_errors:
    #             print(f"- {filepath}: {error_msg}")
    #         self.fail(f"Found {len(all_errors)} errors across scenarios out of {len(scenario_files)} files.")
    #     else:
    #         print("\nAll scenarios ran successfully.")
    def test_agents(self):
        dir = "tests/test_agents"
        print("Testing agents")
        result = run_test(dir)
        print("Testing agents completed")
        self.assertTrue(result == "SUCCESS")

    def test_map(self):
        dir = "tests/test_map"
        print("Tesing map")
        result = run_test(dir)
        print("Testing map completed")
        self.assertTrue(result == "SUCCESS")
    
    def test_automaton(self):
        dir = "tests/test_automaton"
        print("Testing automaton")
        result = run_test(dir)
        print("Testing automaton completed")
        self.assertTrue(result == "SUCCESS")
    
    def test_parser(self):
        dir = "tests/test_parser"
        print("Testing parser")
        result = run_test(dir)
        print("Testing parser completed")
        self.assertTrue(result == "SUCCESS")

    def test_plotter(self):
        dir = "tests/test_plotter"
        print("Testing plotter")
        result = run_test(dir)
        print("Testing plotter completed")
        self.assertTrue(result == "SUCCESS")

    def test_sensor(self):
        dir = "tests/test_sensor"
        print("Testing sensor")
        result = run_test(dir)
        print("Testing sensor completed")
        self.assertTrue(result == "SUCCESS")

    def test_scenario(self):
        dir = "tests/test_scenario"
        print("Testing scenario")
        result = run_test(dir)
        print("Testing scenario completed")
        self.assertTrue(result == "SUCCESS")

if __name__ == "__main__":
    unittest.main()
    
