import unittest
from nf1 import NF1
from nf1 import NF1main


class NF1TestCase(unittest.TestCase):

    def test_nf1(self):
        coms = [(1,2,3), (4,5,6), (7,8,9)]
        gt = [(1,2,3), (4,5,6), (7,8,9)]
        nf = NF1(coms, gt)
        res = nf.summary()
        self.assertEqual(len(res), 2)


    def test_file(self):
        coms = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        with open("test.txt", "w") as o:
            for c in coms:
                o.write("%s\n" % str(c))

        nf = NF1main("test.txt", "test.txt")
        res = nf.summary()
        self.assertEqual(len(res), 2)

    def test_plot(self):
        coms = [(1, 2, 67), (4, 6, 11), (7, 8, 9)]
        gt = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        nf = NF1(coms, gt)
        nf.plot()

if __name__ == '__main__':
    unittest.main()
