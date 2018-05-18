import argparse
from nf1 import NF1main

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"

def execute():
    print("-----------------------------------")
    print("         F1 Communities")
    print("-----------------------------------")
    print("Author: ", __author__)
    print("Email:  ", __contact__)
    print("WWW:    ", __website__)
    print("-----------------------------------")

    parser = argparse.ArgumentParser()

    parser.add_argument('community', type=str,
                            help='the community file')
    parser.add_argument('groundtruth', type=str,
                            help='the ground truth community file')
    parser.add_argument('--plot', metavar='-p', type=str, nargs='?',
                            help='density scatter plot output file', default=False)
    parser.add_argument('--maxpts', metavar='-m', type=int, nargs='?',
                            help='max points to plot', default=None)
    parser.add_argument('--title', metavar='-t', type=str, nargs='?',
                            help='density scatter plot title', default="Precision Recall Scatter Plot")
    parser.add_argument('--outfile', metavar='-o', type=str, nargs='?',
                            help='Output file name', default="plot.png")

    args = parser.parse_args()
    fc = NF1main(args.community, args.groundtruth)

    mean, std, mx, mn, mode = fc.get_f1()
    coverage, gt_coms, id_coms, ratio_coms, node_coverage, redundancy = fc.get_partition_stats()

    quality = fc.get_quality(mean, coverage, redundancy)

    print("          Partition Info          ")
    print("Ground Truth Communities : %d" % gt_coms)
    print("Identified Communities   : %d" % id_coms)
    print("Community Ratio          : %.3f" % ratio_coms)
    print("Ground Truth Matched     : %.3f" % coverage)
    print("Node Coverage            : %.3f" % node_coverage)
    print("-----------------------------------")
    print("      Matching Quality (F1)        ")
    print("Min    : %.3f" % mn)
    print("Max    : %.3f" % mx)
    print("Mode   : %.3f" % mode)
    print("Avg    : %.3f" % mean)
    print("Std    : %.3f" % std)
    print("-----------------------------------")
    print("          Overall Quality          ")
    print("Quality: %.3f" % quality)
    print("-----------------------------------")

    if args.plot is not False:
        fc.plot(max_points=args.maxpts, title=args.title, fileout=args.outfile)
        print("-----------------------------------")
