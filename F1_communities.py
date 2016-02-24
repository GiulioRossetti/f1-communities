import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

__author__ = 'Giulio Rossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


def communities_f1(prl):
    """

    Args:
        prl: list of tuples (precision, recall)

    Returns:
        list of community f1 scores
    """
    f1s = []
    for l in prl:
        x, y = l[0], l[1]
        z = 2 * (x * y) / (x + y)
        z = float("%.2f" % z)
        f1s.append(z)
    return f1s


def f1(f1_list):
    """

    Args:
        f1_list: list of f1 scores

    Returns:
        a tuple composed by (average_f1, std_f1)

    """
    return np.mean(np.array(f1_list)), np.std(np.array(f1_list))


def precision_recall(fc, fgt):
    """

    Args:
        fc: community filename
        fgt: ground truth filename

    Returns:
        a list of tuples (precision, recall)

    """
    f = open(fgt)
    node_to_com = {}
    coms = {}

    cid = 0
    for l in f:
        cid += 1
        try:
            l = l.replace("]", "").replace(" ","").split("[")[1]
        except:
            pass
        ns = map(int, l.rstrip().replace(" ", "\t").replace(",", "\t").split("\t"))
        coms[cid] = ns
        for n in ns:
            node_to_com[n] = cid

    f = open(fc)
    prl = []

    for l in f:
        l = l.replace("]", "").replace(" ","").split("[")[1]
        ns = map(int, l.rstrip().replace(" ", "\t").replace(",", "\t").split("\t"))
        ids = {}

        for n in ns:
            try:
                idd = node_to_com[n]
                if idd not in ids:
                    ids[idd] = 1
                else:
                    ids[idd] += 1
            except KeyError:
                pass
        try:
            label, p = max(ids.iteritems(), key=operator.itemgetter(1))
            prl.append((float(p) / sum(ids.values()), float(len(coms[label]) - p) / len(coms[label])))
        except (ZeroDivisionError, ValueError):
            pass
    return prl


def plot_scatter(prl, max_points=None, fileout="PR_scatter.png", title="Precision Recall Scatter Plot"):
    """

    Args:
        prl: list of tuples (precision, recall)
        max_points: max number of tuples to plot
        fileout: output file
        title: plot title
    """
    prs = [i[0] for i in prl]
    recs = [i[1] for i in prl]

    if max_points is not None:
        prs = prs[:max_points]
        recs = recs[:max_points]

    xy = np.vstack([prs, recs])
    z = gaussian_kde(xy)(xy)

    x = np.array(prs)
    y = np.array(recs)

    base = min(z)
    rg = max(z) - base

    z = np.array(z)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], (z[idx] - base) / rg

    fig, ax = plt.subplots()
    sca = ax.scatter(x, y, c=z, s=60, edgecolor='', cmap='YlOrRd')
    fig.colorbar(sca)
    plt.ylabel("Recall", fontsize=20, labelpad=15)
    plt.xlabel("Precision", fontsize=20)
    plt.ylim([-0.1, 1.1])
    plt.xlim([0, 1.1])
    plt.title(title)
    if matplotlib.get_backend().lower() in ['agg', 'macosx']:
        fig.set_tight_layout(True)
    else:
        fig.tight_layout()

    plt.savefig("%s" % fileout)


if __name__ == "__main__":
    import argparse

    print "-----------------------------------"
    print "         F1 Communities"
    print "-----------------------------------"
    print "Author: ", __author__
    print "Email:  ", __email__
    print "-----------------------------------\n"

    parser = argparse.ArgumentParser()

    parser.add_argument('community', metavar='-c', type=str, nargs='+',
                        help='the community file')
    parser.add_argument('groundtruth', metavar='-gt', type=str, nargs='+',
                        help='the ground truth community file')
    parser.add_argument('--plot', metavar='-p', type=str, nargs='?',
                        help='density scatter plot output file', default=False)
    parser.add_argument('--maxpts', metavar='-m', type=int, nargs='?',
                        help='max points to plot', default=None)
    parser.add_argument('--title', metavar='-t', type=str, nargs='?',
                        help='density scatter plot title', default="Precision Recall Scatter Plot")

    args = parser.parse_args()
    print args

    pr_list = precision_recall(args.community[0], args.groundtruth[0])
    m, std = f1(communities_f1(pr_list))
    print "Mean F1: %.2f\nF1 Std: %.2f" % (float(m), float(std))

    if args.plot is not False:
        plot_scatter(pr_list, max_points=args.maxpts, fileout=args.plot, title=args.title)