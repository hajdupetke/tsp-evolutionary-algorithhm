
def population_split(p, r):
    """
    Calculate how many individuals stay and how many children are created.
    """
    num_pairs = round(r * p / 2)
    num_children = 2 * num_pairs
    keep_n = p - num_children

    assert keep_n >= 0

    return keep_n, num_pairs