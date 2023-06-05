from abstractions.data_structures import Clustering


def gmd(R, S, fm, fs):
    """
    Args:
        R (list of list): the partition, output of the entity resolution that we want to evaluate
        S (list of list): the gold standard
        fm(x,y) -> int (function): cost of merging a group of size x with another group of size y
        fs(x,y) -> int (function): cost of splitting a group into 2 groups of respective sizes x and y
    Returns:
        double : the generalized merge distance between R and S
    """
    # build a map M from record to cluster number
    # store sizes of each cluster in Rsizes
    Rsizes = {}
    M = {}
    for i, group in enumerate(R):
        for r, rec in enumerate(group):
            M[rec] = i
        Rsizes[i] = len(group)
    split_cost = 0
    merge_cost = 0
    for i, group in enumerate(S):
        # determine which clusters in R contain the records in group i
        pMap = {}
        for r, rec in enumerate(group):
            # if we haven't seen the R cluster corresponding to this element we add it to the map
            try:
                M[rec]
            except KeyError as err:
                raise KeyError(
                    'The element of R : {} isn\'t present in S. Check that you did reconcile R and S'.format(err))
            if M[rec] not in pMap:
                pMap[M[rec]] = 0
            # increment the count for this partition
            pMap[M[rec]] += 1

        # compute cost to generate group i of S
        totalRecs = 0
        s_cost = 0
        m_cost = 0
        for i, count in pMap.items():
            if Rsizes[i] > count:
                # add the cost to split R[i]
                s_cost += fs(count, Rsizes[i] - count)
            Rsizes[i] -= count
            if totalRecs != 0:
                # cost to merge into S[i]
                m_cost += fm(count, totalRecs)
            totalRecs += count
        split_cost += s_cost
        merge_cost += m_cost
    return split_cost + merge_cost


def basic_merge_distance(source: Clustering, target: Clustering) -> float:
    bmd = []
    return gmd(source.clustered_rows, target.clustered_rows, lambda *_:1, lambda *_: 1)
