import pyarrow as pa

NUM_SIMULATIONS = 100
PARALLEL_THREADS = 10
INSTANCES = [200, 2000, 10000]
SEED = 0
PATIENCE = [0, 15]


def shd(estimated, true):
    assert set(estimated.nodes()) == set(true.nodes())
    shd_value = 0

    estimated_arcs = set(estimated.arcs())
    true_arcs = set(true.arcs())

    for est_arc in estimated_arcs:
        if est_arc not in true.arcs():
            shd_value += 1
            s, d = est_arc
            if (d, s) in true_arcs:
                true_arcs.remove((d, s))

    for true_arc in true_arcs:
        if true_arc not in estimated_arcs:
            shd_value += 1

    return shd_value


def hamming(estimated, true):
    """
    Calculate the Hamming distance between two directed graphs.

    The Hamming distance is defined as the number of arcs that are present in one graph but not the other,
    considering both directions of the arcs.

    Parameters:
    estimated (networkx.DiGraph): The estimated directed graph.
    true (networkx.DiGraph): The true directed graph.

    Returns:
    int: The Hamming distance between the estimated and true graphs.

    Raises:
    AssertionError: If the set of nodes in the estimated graph does not match the set of nodes in the true graph.
    """
    assert set(estimated.nodes()) == set(true.nodes())
    hamming_value = 0

    estimated_arcs = set(estimated.arcs())
    true_arcs = set(true.arcs())

    for est_arc in estimated_arcs:
        s, d = est_arc
        if (s, d) not in true_arcs and (d, s) not in true_arcs:
            hamming_value += 1

    for true_arc in true_arcs:
        s, d = true_arc
        if (s, d) not in estimated_arcs and (d, s) not in estimated_arcs:
            hamming_value += 1

    return hamming_value


def hamming_type(estimated, true):
    assert set(estimated.nodes()) == set(true.nodes())
    hamming_value = 0

    for n in true.nodes():
        if estimated.node_type(n) != true.node_type(n):
            hamming_value += 1

    return hamming_value
