
def roulette_wheel_selection(costs, rng):
    """
    Fitness-proportionate (roulette wheel) selection; return its INDEX.

    Mitchell Ch. 9 pseudocode, Pr(h_i) = Fitness(h_i) / sum_j Fitness(h_j),
    with Fitness = 1 / cost (report Eq. 5). Smaller cost -> higher fitness ->
    proportionally higher selection probability. One independent spin per
    call (i.e. sampling WITH replacement), exactly as the textbook's
    "probabilistically select" steps imply — the best individual can be lost.

    :param costs: list of tour costs, parallel to the population
    :param rng: random.Random instance
    :return: index into the population
    """
    # Fitness = 1 / cost; costs are strictly positive tour lengths.
    total = 0.0
    fitnesses = []
    for c in costs:
        f = 1.0 / c if c > 0 else 1.0
        fitnesses.append(f)
        total += f

    # Single roulette spin.
    spin = rng.random() * total
    cumulative = 0.0
    for i, f in enumerate(fitnesses):
        cumulative += f
        if spin < cumulative:
            return i
    # Fallback for float rounding: last individual.
    return len(costs) - 1


def tournament_selection(costs, k, rng):
    """
    Select one individual by tournament and return its INDEX.
 
    Picks k random indices and returns the one whose cost is smallest.
    Smallest cost = highest fitness, since fitness is 1/cost, so there is no
    need to divide anything here.
 
    :param costs: list of tour costs, parallel to the population
    :param k: tournament size (selection pressure — larger means more pressure)
    :param rng: random.Random instance
    :return: index into the population
    """
    # Pick k random indices
    indices = [rng.randrange(len(costs)) for _ in range(k)]
    
    # Return the index with the lowest cost
    return min(indices, key=lambda i: costs[i])


def order_crossover(p1, p2, rng):
    """
    Order Crossover (OX). Returns ONE child.
 
    Called twice with the parents swapped, per Algorithm 1.
    The middle segment is copied from p1; the remaining positions are filled
    with the cities of p2 that are not already in the child, in the order they
    appear in p2, starting after the second cut and wrapping around.
 
    :param p1: first parent tour
    :param p2: second parent tour
    :param rng: random.Random instance
    :return: a new child tour (valid permutation)
    """
    
    n = len(p1)
    
    cut1 = rng.randint(0, n-2)
    cut2 = rng.randint(cut1+1, n-1)
    
    child = [None] * n
    child[cut1:cut2+1] = p1[cut1:cut2+1]
    
    used= set(child[cut1:cut2+1])
    
    fill=[]
    for city in p2[cut2+1:] + p2[:cut2+1]:
        if city not in used:
            fill.append(city)
    
    positions = list(range(cut2+1,n))+list(range(0,cut1))
    
    j = 0
    for i in list(positions):
        child[i] = fill[j]
        j += 1
        
    return child


def swap_mutation(tour, rng):
    """
    Swap mutation — exchange the cities at two random positions.
 
    This is the operator named in Algorithm 1, so it is the default.
 
    :param tour: a valid tour
    :param rng: random.Random instance
    :return: a NEW tour, the input is not modified
    """
    
    # Copy the tour
    new_tour = tour.copy()
    
    # Pick two random positions
    i,j = rng.sample(range(len(tour)),2)
    
    # Exchange the two cities in the copy
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

    # Return the copy
    return new_tour
    
    
def inversion_mutation(tour, rng):
    """
    Inversion mutation — reverse the segment between two random positions.
 
    On a symmetric TSP this changes only two edges, while swap changes up to
    four, so it is less disruptive. It is the same move as a 2-opt step, but
    random: it does not check whether the tour got shorter.
 
    Not part of Algorithm 1. Used only for the mutation comparison experiment.
 
    :param tour: a valid tour
    :param rng: random.Random instance
    :return: a NEW tour, the input is not modified
    """
    # Copy the tour
    new_tour = tour.copy()
    
    # Pick two random positions
    i, j = sorted(rng.sample(range(len(tour)), 2))
    
    # Reverse the part between them
    new_tour[i:j+1] = reversed(new_tour[i:j+1])

    # Return the copy
    return new_tour