import time

from src.operators import (
    tournament_selection,
    roulette_wheel_selection,
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


def run_ga(D, params, rng, init_fn, cost_fn, verbose=False, log_interval=100):
    """Run the genetic algorithm (Report Algorithm 1 / Mitchell Ch. 9 Fig. 9.1).

    - P <- init_fn(n, p, rng): Pierre's text says p random permutations;
      Peter Sec. 2.3 replaces this with informed tours (smart_population).
      init_fn injectable so Exp. 1 can compare both.
    - Fitness F = 1/C (Eq. 5). BOTH selection steps are probabilistic with
      the SAME operator, as the pseudocode requires (steps 1+2 share Pr(h_i)):
      default "tournament" (k=5, tutoring-approved operator swap — roulette
      on 1/L has almost no pressure on TSP since tour lengths cluster, e.g.
      best/worst probability ratio ~1.06 late in a run), opt-in "roulette"
      gives the exact textbook Pr(h_i) = F(h_i)/sum_j F(h_j). Sampling is
      WITH replacement (one independent draw per pick), so the best
      individual can be lost — NO elitism, per Mitchell/whiteboard.
      The previous code's bug was deterministic top-(1-r)p elites here.
    - Ps <- (1-r)p probabilistic survivors + 2*(rp/2) OX children (2 per pair).
    - SwapMutation applied to fraction m of Ps. Best-ever tracked and returned.
    - Termination: fixed G generations instead of the textbook's
      fitness_threshold (threshold needs a known target tour length, which
      the algorithm would not know in general).
    - verbose: print init + every log_interval generations (flush=True).
    """
    
    # Get the settings from params
    n = len(D)
    p = params["p"]
    r = params["r"]
    m = params["m"]
    G = params["G"]
    k = params.get("k", 5)
    selection = params.get("selection", "tournament")
    if selection not in ("roulette", "tournament"):
        raise ValueError(f"Unknown selection={selection!r}, expected 'roulette' or 'tournament'")

    # Choose which mutation to use
    if params["mutation"] == "inversion":
        mutation_fn = inversion_mutation
    else:
        mutation_fn = swap_mutation
        
    # Work out how many survive and how many children we need
    keep_n, num_pairs = population_split(p, r)
    num_mutants = int(m * p)

    start = time.time()
    if verbose:
        print(f"  [GA] init: n={n} p={p} r={r} m={m} G={G} selection={selection} k={k} ...", flush=True)

    # Create the first population and calculate its costs
    population = init_fn(n, p, rng)
    costs = [cost_fn(tour, D) for tour in population]
    if verbose:
        print(f"  [GA] init done in {time.time()-start:.1f}s, init_best={min(costs):.1f}", flush=True)

    # Save the best tour found so far
    best_index = min(range(p), key=lambda i: costs[i])
    best_tour = population[best_index][:]
    best_cost = costs[best_index]

    history_best = []
    history_gen = []

    # Repeat for each generation
    for g in range(G):
        # Single selection operator for BOTH textbook steps, so survivors
        # and parents always agree with each other and with the report.
        def select_one():
            if selection == "tournament":
                return tournament_selection(costs, k, rng)
            return roulette_wheel_selection(costs, rng)

        # Step 1 — survivors: probabilistically select (1-r)p members of P
        # with the shared operator, WITH replacement. No elitism: the best
        # individual is NOT guaranteed to survive, per Mitchell/whiteboard.
        new_population = [population[select_one()][:] for _ in range(keep_n)]

        # Step 2 — crossover: probabilistically select r*p/2 pairs from P
        # via the same Pr(h_i), two OX children per pair (roles swapped).
        for _ in range(num_pairs):
            i = select_one()
            j = select_one()

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
        if verbose and ((g + 1) % log_interval == 0 or g + 1 == G):
            elapsed = time.time() - start
            print(f"  [GA] gen {g+1}/{G} best={best_cost:.1f} gen_best={generation_best:.1f} elapsed={elapsed:.1f}s",
                  flush=True)

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