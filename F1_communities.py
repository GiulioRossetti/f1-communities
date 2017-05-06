import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"


class F1_communities(object):
    """

    """

    def __init__(self, community_filename, groundtruth_filename):
        """

        :param community_filename: community filename
        :param groundtruth_filename: groundtruth community filename
        :return:
        """
        self.community_filename = community_filename
        self.ground_truth_filename = groundtruth_filename
        self.matched_gt = {}
        self.gt_count = 0
        self.id_count = 0
        self.gt_nodes = {}
        self.id_nodes = {}

    def get_f1(self, prl):
        """

        :param prl: list of tuples (precision, recall)
        :return: a tuple composed by (average_f1, std_f1)
        """

        f1_list = np.array(self.__communities_f1(prl))

        return (np.mean(f1_list), np.std(f1_list),
                np.max(f1_list), np.min(f1_list),
                scipy.stats.mode(f1_list)[0][0])

    def get_partition_stats(self):
        return (float(len(self.matched_gt))/float(self.gt_count),  # coverage
                self.gt_count,
                self.id_count,
                float(self.id_count)/float(self.gt_count),
                float(len(self.id_nodes))/float(len(self.gt_nodes)),
                float(self.id_count)/float(len(self.matched_gt))  # redundancy
                )

    def get_precision_recall(self):
        """

        :return: a list of tuples (precision, recall)
        """

        f = open(self.ground_truth_filename)
        node_to_com = {}
        coms = {}

        # read ground truth
        cid = 0
        for l in f:
            cid += 1
            try:
                l = l.replace("]", "").replace(" ", "").split("[")[1]
            except:
                pass
            ns = map(int, l.rstrip().replace(" ", "\t").replace(",", "\t").split("\t"))

            if len(ns) == 2:
                raise Exception

            coms[cid] = ns
            for n in ns:
                if n not in self.gt_nodes:
                    self.gt_nodes[n] = True

                if n not in node_to_com:
                    node_to_com[n] = [cid]
                else:
                    node_to_com[n].append(cid)

        self.gt_count = cid

        f = open(self.community_filename)
        prl = []

        # match each community
        idc = 0
        for l in f:
            try:
                idc += 1
                l = l.replace("]", "").replace(" ", "").split("[")[1]
            except:
                pass

            ns = map(int, l.rstrip().replace(" ", "\t").replace(",", "\t").split("\t"))
            ids = {}

            for n in ns:
                if n not in self.id_nodes:
                    self.id_nodes[n] = True

                try:
                    # community in ground truth
                    idd_list = node_to_com[n]
                    for idd in idd_list:
                        if idd not in ids:
                            ids[idd] = 1
                        else:
                            ids[idd] += 1
                except KeyError:
                    pass
            try:
                # identify the maximal match ground truth communities (label) and their absolute frequency (p)
                maximal_match = {label: p for label, p in ids.iteritems() if p == max(ids.values())}

                for label, p in maximal_match.iteritems():
                    if label not in self.matched_gt:
                        self.matched_gt[label] = True

                    precision = float(p)/len(ns)
                    recall = float(p)/len(coms[label])
                    prl.append((precision, recall))
            except (ZeroDivisionError, ValueError):
                pass

        self.id_count = idc
        return prl

    @staticmethod
    def get_quality(f1, coverage, redundancy):
        """
        How much the communities are similar, how much the ground truth is covered
        and how many communities are needed

        :param f1:
        :param coverage:
        :param redundancy:
        :return:
        """
        return (f1 * coverage)/redundancy

    @staticmethod
    def __communities_f1(prl):
        """

        :param prl: list of tuples (precision, recall)
        :return: list of community f1 scores
        """

        f1s = []
        for l in prl:
            x, y = l[0], l[1]
            z = 2 * (x * y) / (x + y)
            z = float("%.2f" % z)
            f1s.append(z)
        return f1s

    @staticmethod
    def __plot_scatter_singular(prl, max_points=None, fileout="PR_scatter.png", title="Precision Recall Scatter Plot"):
        """

        :param prl: list of tuples (precision, recall)
        :param max_points: max number of tuples to plot
        :param fileout: output filename
        :param title: plot title
        """

        prs = [i[0] for i in prl]
        recs = [i[1] for i in prl]

        if max_points is not None:
            prs = prs[:max_points]
            recs = recs[:max_points]

        # histogram definition
        xyrange = [[0, 1],[0, 1]]  # data range
        bins = [50, 50]  # number of bins
        thresh = 3  # density threshold
        xdat, ydat = np.array(prs), np.array(recs)

        # histogram the data
        hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)

        # select points within the histogram
        hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
        plt.imshow(np.flipud(hh.T), cmap='jet',
                   extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', alpha=0.8)
        plt.colorbar()
        plt.title(title)
        plt.ylabel("Recall", fontsize=20, labelpad=15)
        plt.xlabel("Precision", fontsize=20)
        plt.savefig(fileout)

    @staticmethod
    def __plot_scatter(prl, max_points=None, fileout="PR_scatter.png", title="Precision Recall Scatter Plot"):
        """

        :param prl: list of tuples (precision, recall)
        :param max_points: max number of tuples to plot
        :param fileout: output filename
        :param title: plot title
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
        sca = ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet)
        fig.colorbar(sca)
        plt.ylabel("Recall", fontsize=20, labelpad=15)
        plt.xlabel("Precision", fontsize=20)
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, 1.01])
        plt.title(title)
        if matplotlib.get_backend().lower() in ['agg', 'macosx']:
            fig.set_tight_layout(True)
        else:
            fig.tight_layout()

        plt.savefig("%s" % fileout)

    @staticmethod
    def plot(prl, max_points=None, fileout="PR_scatter.png", title="Precision Recall Scatter Plot"):
        """

        :param prl: list of tuples (precision, recall)
        :param max_points: max number of tuples to plot
        :param fileout: output filename
        :param title: plot title
        """

        try:
            fc.__plot_scatter(prl, max_points=max_points, title=title, fileout=fileout)
        except np.linalg.linalg.LinAlgError:
            fc.__plot_scatter_singular(prl, max_points=max_points, title=title, fileout=fileout)
        print "Density Scatter Plot Generated"

    @staticmethod
    def convert_coms(filename):
        """

        :param filename: community filename
        """

        try:
            f = open("%s" % filename)
            o = open("%s_r" % filename, "w")
            com2nodes = {}
            for l in f:
                l = map(int, l.rstrip().replace(" ", "\t").replace(",", "\t").split("\t"))
                if l[1] in com2nodes:
                    com2nodes[l[1]].append(l[0])
                else:
                    com2nodes[l[1]] = [l[0]]
            for l in com2nodes.values():
                l = str(l).strip("[]").replace(" ","")
                o.write("%s\n" % l)
            o.flush()
            o.close()
        except:
            print "File Format Error"

if __name__ == "__main__":
    import argparse
    print "-----------------------------------"
    print "         F1 Communities"
    print "-----------------------------------"
    print "Author: ", __author__
    print "Email:  ", __contact__
    print "WWW:    ", __website__
    print "-----------------------------------"

    parser = argparse.ArgumentParser()

    parser.add_argument('community', type=str,
                        help='the community file')
    parser.add_argument('groundtruth',  type=str,
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
    fc = F1_communities(args.community, args.groundtruth)
    prl_list = None
    try:
        prl_list = fc.get_precision_recall()
    except Exception:
        fc.convert_coms(args.groundtruth)
        fc = F1_communities(args.community, "%s_r" % args.groundtruth)
        prl_list = fc.get_precision_recall()

    mean, std, mx, mn, mode = fc.get_f1(prl_list)
    coverage, gt_coms, id_coms, ratio_coms, node_coverage, redundancy = fc.get_partition_stats()

    quality = fc.get_quality(mean, coverage, redundancy)

    print "          Partition Info          "
    print "Ground Truth Communities : %d" % gt_coms
    print "Identified Communities   : %d" % id_coms
    print "Community Ratio          : %.3f" % ratio_coms
    print "Ground Truth Matched     : %.3f" % coverage
    print "Node Coverage            : %.3f" % node_coverage
    print "-----------------------------------"
    print "      Matching Quality (F1)        "
    print "Min    : %.3f" % mn
    print "Max    : %.3f" % mx
    print "Mode   : %.3f" % mode
    print "Avg    : %.3f" % mean
    print "Std    : %.3f" % std
    print "-----------------------------------"
    print "          Overall Quality          "
    print "Quality: %.3f" % quality
    print "-----------------------------------"

    if args.plot is not False:
        fc.plot(prl_list, max_points=args.maxpts, title=args.title, fileout=args.outfile)
        print "-----------------------------------"
