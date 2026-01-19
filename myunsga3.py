import random
import numpy as np


# Objective 1: Maximize node properties (sum of node weights)
def gamma_diversity(graph, team):
    gama = set()
    for nd in team:
        gama.update(set(graph.nodes[nd[0]]["skills"].split(", ")))
    return len(gama)


def diameter_distance(graph, team) -> float:
    team_nodes = set()
    leader = team[0][0]
    team_nodes.add(leader)
    uniq_members = set([mmbr[0] for mmbr in team])
    for expert in uniq_members:
        for node in nx.dijkstra_path(graph, leader, expert, weight="weight"):
            team_nodes.add(node)
    team_graph = nx.subgraph(graph, team_nodes).copy()
    return nx.diameter(team_graph)


# Objective 2: Minimize the sum of distances between nodes
def sum_distance(graph, team, task) -> float:
    import networkx as nx
    # from Team import Team
    sd = 0
    for skill_i in task:
        for skill_j in task:
            if skill_i != skill_j:
                for member1 in team:
                    expert_i = member1[0]
                    if skill_i == member1[1]:
                        for member2 in team:
                            if skill_j == member2[1]:
                                expert_j = member2[0]
                                if expert_i in graph and expert_j in graph and \
                                        nx.has_path(graph, str(expert_i), str(expert_j)):
                                    sd += nx.dijkstra_path_length(graph, str(expert_i), str(expert_j), weight="weight")
    sd /= 2
    return round(sd, 3)


# Generate an initial population (random node subsets)
def initialize_population(population_size, task):
    initial_population = []
    for _ in range(population_size):
        sol = []
        for skill in task:
            sol.append([np.random.choice(list(popularity[skill]), 1, True)[0], skill])
        initial_population.append(sol)
    return initial_population


# Crossover operation (two-point crossover)
def crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


# Mutation operation (shuffle mutation)
def mutate(individual, mutation_rate=0.1):
    skill_list = []
    for skill in task:
        if len(popularity[skill]) > 1:
            skill_list.append(skill)
    if len(skill_list) > 0:
        choosen_skill = random.choice(skill_list)
        if random.random() < mutation_rate:
            i = -1
            for pt in individual:
                i += 1
                if pt[1] == choosen_skill:
                    individual[i][0] = random.choice(list(popularity[choosen_skill]))
                    break
    return individual


# Fast non-dominated sorting
def non_dominated_sorting(population, fitness_values):
    fronts = [[]]
    domination_counts = [0] * len(population)  # number of solutions dominated by the index solution
    dominated_solutions = [[] for _ in range(len(population))]  # Index list of solutions better than the index solution

    for p in range(len(population)):
        for q in range(len(population)):
            # Initialize dominance flags
            p_dominates_q = False
            q_dominates_p = False
            # Maximization objective: compare for maximization
            if fitness_values[p][0] > fitness_values[q][0]:
                p_dominates_q = True
            elif fitness_values[p][0] < fitness_values[q][0]:
                q_dominates_p = True
            # Minimization objective: compare for minimization
            if fitness_values[p][1] < fitness_values[q][1]:
                p_dominates_q = True
            elif fitness_values[p][1] > fitness_values[q][1]:
                q_dominates_p = True
            # If p dominates q, add q to p's dominance list
            if p_dominates_q and not q_dominates_p:
                dominated_solutions[p].append(q)
            # If q dominates p, increase the domination count of p
            elif q_dominates_p and not p_dominates_q:
                domination_counts[p] += 1
        # If p is non-dominated (i.e., dominated_count[p] == 0), add it to the first front
        if domination_counts[p] == 0:
            fronts[0].append(p)
    # Build subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for p in fronts[current_front]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1  # Decrease the domination count of dominated points
                if domination_counts[q] == 0:
                    next_front.append(q)  # If q becomes non-dominated, add it to the next front
        current_front += 1
        fronts.append(next_front)
    # Remove the last empty front
    if not fronts[-1]:
        fronts.pop()
    return fronts


# Calculate crowding distance
def crowding_distance(population, fitness_values, front):
    distance = [0] * len(fitness_values)
    num_objectives = len(fitness_values[0])

    for i in range(num_objectives):
        sorted_front = sorted(front, key=lambda x: fitness_values[x][i])
        if len(sorted_front) < 3:
            distance[0] = distance[-1] = 0
        else:
            distance[0] = distance[-1] = float('inf')

        max_val = fitness_values[sorted_front[-1]][i]
        min_val = fitness_values[sorted_front[0]][i]

        for j in range(1, len(sorted_front) - 1):
            if max_val - min_val == 0:
                distance[sorted_front[j]] += 0
            else:
                distance[sorted_front[j]] += (fitness_values[sorted_front[j + 1]][i] -
                                              fitness_values[sorted_front[j - 1]][i]) / (max_val - min_val)

    return distance


# Selection based on non-dominated sorting and crowding distance
def selection(population, fitness_values, num_offspring):
    fronts = non_dominated_sorting(population, fitness_values)
    new_population = []
    for front in fronts:
        if len(new_population) + len(front) > num_offspring:
            distances = crowding_distance(population, fitness_values, front)
            sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
            # sorted_front = sorted(front, key=lambda x: distances[x], reverse=True)
            new_population.extend([population[i] for i in sorted_front[:num_offspring - len(new_population)]])
            break
        else:
            new_population.extend([population[i] for i in front])
    return new_population


def unique_teams(listofteams):
    unique_teams = []
    for team in listofteams:
        if team not in unique_teams:
            unique_teams.append(team)
    return unique_teams


def normalize_fitness(fitness_values, n_obj=2):
    """Normalize fitness values using ideal and nadir points"""
    # Simple normalization: subtract min and divide by range
    f_array = np.array(fitness_values)
    f_min = np.min(f_array, axis=0)
    f_max = np.max(f_array, axis=0) + 1e-10  # avoid division by zero
    f_norm = (f_array - f_min) / (f_max - f_min)
    return f_norm


def perpendicular_distance(f_norm, ref_dir):
    """Calculate perpendicular distance from normalized fitness to reference direction"""
    # Projection scalar: dot product of f_norm and ref_dir
    proj_scalar = np.dot(f_norm, ref_dir)
    # Projected point: proj_scalar * ref_dir (since ref_dir is unit vector)
    proj_point = proj_scalar * ref_dir
    # Perpendicular distance (Euclidean distance to projection)
    return euclidean(f_norm, proj_point)


def associate_to_reference(fitness_values, ref_dirs):
    """Associate each solution to closest reference direction"""
    f_norm = normalize_fitness(fitness_values)
    associations = np.full(len(fitness_values), -1, dtype=int)
    niche_counts = np.zeros(len(ref_dirs))

    for i, f_i in enumerate(f_norm):
        min_dist = float('inf')
        closest_ref = -1
        for j, ref_dir in enumerate(ref_dirs):
            dist = perpendicular_distance(f_i, ref_dir)
            if dist < min_dist:
                min_dist = dist
                closest_ref = j
            elif dist == min_dist:
                # Tie-breaking: prefer less populated niche
                if niche_counts[j] < niche_counts[closest_ref]:
                    closest_ref = j
        associations[i] = closest_ref
        niche_counts[closest_ref] += 1

    return associations, niche_counts


def nsga3_reference_selection(population, fitness_values, population_size, ref_dirs):
    """NSGA-III style selection using reference directions"""
    fronts = non_dominated_sorting(population, fitness_values)
    selected = []

    # Fill with complete fronts until population is full or splitting front
    remaining = population_size
    front_idx = 0

    while remaining > 0 and front_idx < len(fronts):
        front_size = len(fronts[front_idx])
        if remaining >= front_size:
            # Take entire front
            selected.extend(fronts[front_idx])
            remaining -= front_size
        else:
            # Splitting front - use reference direction niching
            break
        front_idx += 1

    if remaining > 0 and len(fronts) < front_idx:
        # Need niching selection from splitting front
        splitting_front = fronts[front_idx]
        splitting_fitness = [fitness_values[i] for i in splitting_front]
        splitting_indices = splitting_front  # indices in original population

        # Associate splitting front solutions to reference directions
        associations, niche_counts = associate_to_reference(splitting_fitness, ref_dirs)

        # Sort reference directions by niche count (ascending - prefer empty)
        ref_order = np.argsort(niche_counts)

        while remaining > 0 and len(ref_order) > 0:
            for ref_idx in ref_order:
                if niche_counts[ref_idx] == 0:  # Empty niche first
                    # Find closest solution to this reference
                    candidates = [i for i, assoc in enumerate(associations)
                                  if assoc == ref_idx]
                    if candidates:
                        closest_idx = min(candidates,
                                          key=lambda i: perpendicular_distance(
                                              normalize_fitness([splitting_fitness[i]])[0],
                                              ref_dirs[ref_idx]))
                        selected.append(population[splitting_indices[closest_idx]])
                        remaining -= 1
                        if remaining == 0:
                            break

            # Now fill remaining niches (allow multiple per niche)
            for ref_idx in ref_order:
                candidates = [i for i, assoc in enumerate(associations)
                              if assoc == ref_idx and splitting_indices[i] not in selected]
                if candidates and remaining > 0:
                    # Random selection among remaining candidates for this niche
                    cand_idx = random.choice(candidates)
                    selected.append(population[splitting_indices[cand_idx]])
                    remaining -= 1
                    if remaining == 0:
                        break

    return selected[:population_size]


def nsga3(population_size, num_generations, crossover_prob, mutation_prob, task, ref_dirs):
    """
    NSGA-III implementation using reference directions for selection.
    Assumes ref_dirs is a numpy array of shape (n_ref_dirs, n_objectives)
    where each row is a unit vector reference direction.
    """
    import time
    start = time.time_ns()
    population = initialize_population(population_size, task)
    population = unique_teams(population)

    for generation in range(num_generations):
        offspring = []
        while len(offspring) < population_size:
            if len(population) < 2:
                fitness_values = [(gamma_diversity(graph, ind),
                                   diameter_distance(graph, ind)) for ind in population]
                return population, time.time_ns() - start, fitness_values
            # parent1, parent2 = random.sample(population, 2)
            sorted_population = sorted(population, key=lambda x: min(diameter_distance(graph,x), gamma_diversity(graph, x)))
            parent1 = sorted_population[0]
            parent2 = sorted_population[1]

            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)
            offspring.extend([child1, child2])

        population.extend(offspring[:population_size])
        population = unique_teams(population)
        fitness_values = [(gamma_diversity(graph, ind),
                           diameter_distance(graph, ind)) for ind in population]

        # NSGA-III selection using reference directions
        population_lst = nsga3_reference_selection(population, fitness_values,
                                                   population_size, ref_dirs)
        for i in range(len(population_lst)):
            population.append(population[i])
    # Final front extraction
    fitness_values = [(gamma_diversity(graph, ind),
                       diameter_distance(graph, ind)) for ind in population]
    fronts = non_dominated_sorting(population, fitness_values)
    final = [population[i] for i in fronts[0]]
    final_fitness = [(gamma_diversity(graph, ind),
                      diameter_distance(graph, ind)) for ind in final]

    return final, time.time_ns() - start, final_fitness


# Run the NSGA-II algorithm
population_size = 20
num_generations = 25
crossover_prob = 0.5
mutation_prob = 0.5

import networkx as nx
from pymoo.util.ref_dirs import get_reference_directions

# networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
#             "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
networks = ["vldb"]
for network in networks:
    print(network)
    tasks = []
    graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
    with open("/home/ramesh/dblp/input/" + network + "_tasks.txt") as file:
        for line in file:
            task = [x.strip() for x in line.strip("\n").split("\t") if x]
            tasks.append(task)
    popularity = dict()  # experts_for_skill i.e. skill:list of experts
    for node in graph.nodes:
        if "skills" in graph.nodes[node]:
            for skill in graph.nodes[node]["skills"].split(","):
                if len(skill) > 0:
                    if skill in popularity:
                        popularity[skill.strip()].add(node)
                    else:
                        popularity[skill.strip()] = set()
                        popularity[skill.strip()].add(node)
    open("/home/ramesh/dblp/output/" + network + "_nsga3_6_teams_all.txt", "w").close()
    i = 0
    for task in tasks:
        i += 1
        ref_dirs = get_reference_directions("energy", 2, 100, n_partitions=6)
        final_population, ptime, final_fitness = nsga3(population_size, num_generations, crossover_prob,
                                                       mutation_prob, task, ref_dirs)
        with open("/home/ramesh/dblp/output/" + network + "_nsga3_6_teams_all.txt", "a") as file:
            for team in final_population:
                print(team)
                # Ensure that each element in team is a pair
                formatted_team = "\t".join(f'{pair[0]}\t{pair[1]}' for pair in team if len(pair) == 2)
                file.write(f"{ptime}\t{formatted_team}\n")
            if i % 5 == 0:
                file.write("\n")
        with open("/home/ramesh/dblp/output/" + network + "_nsga3_6_fitness.txt", "a") as file1:
            for tmvl in final_fitness:
                file1.write(f"{tmvl[0]}\t{tmvl[1]}\n")
            # import matplotlib.pyplot as plt
            # # Output the final solutions
            # for ind, fitness in zip(final_population, final_fitness):
            #     print(f"Selected Nodes: {ind}, Fitness (Max property, Min distance): {fitness}")
            #     # Plot results
            #     plt.scatter([ind[1] for ind in final_fitness], [ind[0] for ind in final_fitness])
            # plt.xlabel('Communication cost')
            # plt.ylabel('Gamma diversity')
            # plt.title('NSGA-II Results')
            # plt.show()


def analysis():
    import networkx as nx
    networks = ["vldb"]
    # networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
    #             "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        print(network)
        open("/home/ramesh/dblp/output/" + network + "_" + "nsga3_6_teams.txt", "w").close()
        ab = set()
        with open("/home/ramesh/dblp/output/" + network + "_nsga3_6_teams_all.txt") as file:
            i = 0
            ptime = 0
            for line in file:
                words = line.strip("\n").split("\t")
                task = {words[i] for i in range(2, len(words), 2)}
                # experts = {words[i] for i in range(1,len(words),2)}
                team = [(words[i], words[i + 1]) for i in range(1, len(words), 2)]
                with open("/home/ramesh/dblp/output/" + network + "_nsga3_6_teams.txt", "a") as wfile:
                    if len(words) > 1 and words[0] not in ab:
                        i += 1
                        ab.add(words[0])
                        wfile.write(words[0] + "\t" + words[1] + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
                        ptime += int(words[0])
                        if i % 5 == 0:
                            with open("/home/ramesh/dblp/output/" + network + "_nsga3_6_analysis.txt", "a") as w2file:
                                w2file.write(str(len(task)) + "\t" + str(ptime / 5) + "\n")


# def comparison():
#     networks = ["icdt"]
#     for network in networks:
#         print(network)
#         aco_teams = []
#         with open("/home/ramesh/dblp/output/" + network + "_popular_teams.txt") as file:
#             for line in file:
#                 result = [x for x in line.strip("\n").split("\t") if x]
#                 team = []
#                 for k in range(len(result)):
#                     if k > 1 and k % 2 == 0:
#                         team.append([result[k], result[k + 1]])
#                 aco_teams.append(team)
#         nsga3_teams = []
#         with open("/home/ramesh/dblp/output/" + network + "_nsga3_teams.txt") as file:
#             for line in file:
#                 result = line.strip("\n").split("\t")
#                 if len(result) > 1:
#                     team = [[result[k], result[k + 1]] for k in range(1,len(result),2)]
#                     for team_aco in aco_teams:
#                         if len(team_aco)>len(team):
#                             break
#                         if sorted(team) == sorted(team_aco):
#                             nsga3_teams.append(team)
#                             print(len(team),team)


#
if __name__ == '__main__':
    analysis()
    # comparison()