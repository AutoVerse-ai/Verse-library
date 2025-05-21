import unittest

from verse.analysis import AnalysisTree, AnalysisTreeNode

def collect_id_from_node(tree: AnalysisTree) -> set[int]:
    res = set()
    for node in tree.nodes:
        res.add(node.id)
    return res
class TestTreeNode(unittest.TestCase):
    #TODO: Need some discussion about what to test here
    def test_id(self):
        tree = AnalysisTreeNode()
        self.assertTrue(True)

    # TODO:

if __name__ == "__main__":
    unittest.main()