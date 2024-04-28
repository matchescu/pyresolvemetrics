import json

from matchescu.metrics import fsm, algebraic


def _load_json(file_path: str):
    with open(file_path) as f:
        return json.load(f)


def _make_hashable(item: list) -> tuple:
    return tuple(item)


def _hashable_pair(pair) -> tuple[tuple, ...]:
    return _make_hashable(pair[0]), _make_hashable(pair[1])


def _extract_cluster(cluster) -> set[tuple]:
    return set(_make_hashable(item) for item in cluster)


if __name__ == "__main__":
    gt = _load_json("../data/mini-buy/gt.json")
    er = _load_json("../data/mini-buy/results/0.79.json")

    gt_pairs = list(_hashable_pair(pair) for pair in gt["fsm"])
    er_pairs = list(_hashable_pair(pair) for pair in er["fsm"])

    print("precision", fsm.precision(gt_pairs, er_pairs))
    print("recall", fsm.recall(gt_pairs, er_pairs))
    print("F1", fsm.f1(gt_pairs, er_pairs))

    gt_partition = list(_extract_cluster(c) for c in gt["algebraic"])
    er_partition = list(_extract_cluster(c) for c in er["algebraic"])

    print("pairwise precision", algebraic.pair_precision(gt_partition, er_partition))
    print("pairwise recall", algebraic.pair_recall(gt_partition, er_partition))
    print("pairwise F1", algebraic.pair_comparison_measure(gt_partition, er_partition))

    print("cluster precision", algebraic.cluster_precision(gt_partition, er_partition))
    print("cluster recall", algebraic.cluster_recall(gt_partition, er_partition))
    print(
        "cluster F1", algebraic.cluster_comparison_measure(gt_partition, er_partition)
    )

    print("Rand index", algebraic.rand_index(gt_partition, er_partition))
    print(
        "adjusted Rand index", algebraic.adjusted_rand_index(gt_partition, er_partition)
    )
    print("Talburt-Wang index", algebraic.twi(gt_partition, er_partition))
