import time

from src.operators import (
    tournament_selection,
    order_crossover,
    swap_mutation,
    inversion_mutation,
)


def population_split(p, r):
    """
    Calculate how many individuals stay and how many children are created.
    """
    num_pairs = round(r * p / 2)
    num_children = 2 * num_pairs
    keep_n = p - num_children

    assert keep_n >= 0

    return keep_n, num_pairs


def run_ga(D, params, rng, init_fn, cost_fn):
    """Run the genetic algorithm."""
    
    # Get the settings from params
    n = len(D)
    p = params["p"]
    r = params["r"]
    m = params["m"]
    G = params["G"]
    k = params["k"]

    # Choose which mutation to use
    if params["mutation"] == "inversion":
        mutation_fn = inversion_mutation
    else:
        mutation_fn = swap_mutation
        
    # Work out how many survive and how many children we need
    keep_n, num_pairs = population_split(p, r)
    num_mutants = int(m * p)

    start = time.time()

    # Create the first population and calculate its costs
    population = init_fn(n, p, rng)
    costs = [cost_fn(tour, D) for tour in population]

    # Save the best tour found so far
    best_index = min(range(p), key=lambda i: costs[i])
    best_tour = population[best_index][:]
    best_cost = costs[best_index]

    history_best = []
    history_gen = []

    # Repeat for each generation
    for g in range(G):
        new_population = []

        # Select survivors using tournament selection
        for _ in range(keep_n):
            i = tournament_selection(costs, k, rng)
            new_population.append(population[i][:])

        # Create children using crossover
        for _ in range(num_pairs):
            i = tournament_selection(costs, k, rng)
            j = tournament_selection(costs, k, rng)

            child1 = order_crossover(population[i], population[j], rng)
            child2 = order_crossover(population[j], population[i], rng)

            new_population.append(child1)
            new_population.append(child2)

        # Mutate a fixed number of individuals
        positions = rng.sample(range(p), num_mutants)

        for i in positions:
            new_population[i] = mutation_fn(new_population[i], rng)

        # Replace the old population and calculate the new costs
        population = new_population
        costs = [cost_fn(tour, D) for tour in population]

        # Find the best tour in this generation
        generation_best = min(costs)

        if generation_best < best_cost:
            best_index = costs.index(generation_best)
            best_cost = generation_best
            best_tour = population[best_index][:]

        # Save both the current best and the best ever found
        history_gen.append(generation_best)
        history_best.append(best_cost)

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "history_best": history_best,
        "history_gen": history_gen,
        "seconds": time.time() - start,
    }


def random_search(D, n, evaluations, rng, cost_fn):
    """Generate random tours and keep the best one."""
    start = time.time()

    best_tour = None
    best_cost = float("inf")
    history_best = []
    history_gen = []

    for i in range(evaluations):
        tour = list(range(n))
        rng.shuffle(tour)

        cost = cost_fn(tour, D)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]

        history_best.append(best_cost)
        history_gen.append(best_cost)

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "history_best": history_best,
        "history_gen": history_gen,
        "seconds": time.time() - start,
    }