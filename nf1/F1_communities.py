from nf1.NF1 import NF1

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"


class F1_communities(NF1):
    """

    """

    def __init__(self, community_filename, groundtruth_filename):
        """

        :param community_filename: community filename
        :param groundtruth_filename: groundtruth community filename
        :return:
        """

        coms = self.__read_coms(community_filename)
        gtc = self.__read_coms(groundtruth_filename)
        super(self.__class__, self).__init__(coms, gtc)

    def __read_coms(self, filename):
        com = []
        with open(filename) as f:
            for l in f:
                try:
                    l = l.replace("(","[").replace(")","]").replace("]", "").replace(" ", "").split("[")[1]
                except:
                    pass
                ns = tuple(map(int, l.rstrip().replace(" ", "\t").replace(",", "\t").split("\t")))

                if len(ns) == 2:
                    raise Exception
                com.append(ns)
        return com
