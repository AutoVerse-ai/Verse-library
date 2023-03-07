from verse.analysis.analysis_tree import AnalysisTree

tree1 = AnalysisTree.load('demo/tacas2023/exp1/output1.json')
tree2 = AnalysisTree.load('demo/tacas2023/exp1/output1_arch.json')

print("tree1 contains tree2?", tree1.contains(tree2))
print("tree2 contains tree1?", tree2.contains(tree1))