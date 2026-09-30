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
    for i in list(range(cut2+1, n)) + list(range(cut1)):
        child[i] = fill[j]
        j += 1
        
    return child