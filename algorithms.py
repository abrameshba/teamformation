import sys
import time

import networkx as nx


def blenddtfp(graph, task, popularity_skill):
    '''
    Blend value of skill based diverse team formation problem
    :return:
    '''
    start = time.time_ns()
    uskills = set()  # Universal set of skills
    expertise = {node: set() for node in graph.nodes}  # Dictionary:expert(key):list of skills(value) as strings
    for node in list(graph.nodes):
        if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
            skls = set(graph.nodes[node]["skills"].split(","))
            expertise[node] = skls
            uskills.update(skls)
    bs = {skill: set() for skill in uskills}  # Blend value of skills, dictionary
    support = {skill: set() for skill in uskills}
    for node in list(graph.nodes):
        if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
            skls = set(graph.nodes[node]["skills"].split(","))
            for skl in skls:
                bs[skl].update(skls)
                support[skl].add((node))
    teams = []
    dbs = bs.copy()
    lbs = min([(skl, bs[skl]) for skl in task], key=lambda x: len(x[1]))
    del dbs[lbs[0]]
    for expert in support[lbs[0]]:
        dtask = task.copy()
        team = []
        Tc = []  # Task(skills) covered
        team.append((expert, ""))
        for skill in set(task).intersection(expertise[expert]):
            team.append((expert, skill))
            dtask.remove(skill)
            Tc.append(skill)
        while (len(Tc) != len(task)):
            skilld = min([(skl, bs[skl]) for skl in dtask], key=lambda x: len(x[1]))
            cskls = []
            for exprt in list(support[skilld[0]]):
                cskls.append((exprt, (len(expertise[exprt]) / nx.dijkstra_path_length(graph, expert, exprt))))
            cv = max(cskls, key=lambda x: x[1])
            for skill in set(set(task).difference(Tc)).intersection(expertise[cv[0]]):
                team.append((cv[0], skill))
                dtask.remove(skill)
                Tc.append(skill)
        teams.append(team)
    current_inv_ginisimpson_div = 0
    best_team = []
    for team in teams:
        inv_ginisimpson_div = inverse_gini_simpson_diversisty(graph, team, task)
        if current_inv_ginisimpson_div < inv_ginisimpson_div:
            best_team = team
    return best_team, time.time_ns() - start


def pplrtdtfp(graph, task):
    '''
    Popularity of skill based diverse team formation problem
    :return:
    '''
    start = time.time_ns()
    uskills = set()
    expertise = {node: set() for node in graph.nodes}
    for node in list(graph.nodes):
        if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
            skls = set(graph.nodes[node]["skills"].split(","))
            expertise[node] = skls
            uskills.update(skls)
    support = {skill: set() for skill in uskills}
    for node in list(graph.nodes):
        if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
            skls = set(graph.nodes[node]["skills"].split(","))
            for skl in skls:
                support[skl].add((node))
    teams = []
    lps = min([(skl, support[skl]) for skl in task], key=lambda x: len(x[1]))
    for expert in support[lps[0]]:
        dtask = task.copy()
        team = []
        Tc = []
        Tc.append(lps[0])
        team.append(expert)
        dtask.remove(lps[0])
        while (len(Tc) != len(task)):
            clbs = min([(skl, support[skl]) for skl in dtask], key=lambda x: len(x[1]))
            cskls = []
            for exprt in list(support[clbs[0]]):
                cskls.append((exprt, expertise[exprt]))
            cv = max(cskls, key=lambda x: len(x[1]))
            team.append((cv[0]))
            Tc.append(expertise[cv[0]].intersection(set(dtask)))
            dtask.remove(clbs[0])
        teams.append(team)
    cgama = 0
    best_team = []
    for team in teams:
        task_size, gama_div, diameter_d, sum_distnc, Inv_ginisimpson_div, shannon_div = fitness(graph, team, task)
        if cgama < gama_div:
            best_team = team
    return fitness(graph, best_team, task), time.time_ns() - start, best_team


def gamma_diversity(graph, team, task):
    gama = set()
    for nd in team:
        gama.update(set(graph.nodes[nd[0]]["skills"].split(",")))
    return len(gama)


def shannon_diversity(graph, team, task):
    total_skills = dict()
    for node in team:
        if len(graph.nodes[node[0]]) > 0 and "skills" in graph.nodes[node[0]]:
            skls = set(graph.nodes[node[0]]["skills"].split(","))
            for skill in skls:
                if skill not in total_skills:
                    total_skills[skill] = set()
                total_skills[skill].add(node[0])
    shnn_sum = 0
    import math
    for skill in total_skills.keys():
        shnn_sum += (len(total_skills[skill]) / len(team) * math.log(len(total_skills[skill]) / len(team)))
    return -1 * shnn_sum


def inverse_gini_simpson_diversisty(graph, team, task):
    total_skills = dict()
    import numpy as np
    uniq_members = set([mmbr[0] for mmbr in team])
    for node in team:
        if len(graph.nodes[node[0]]) > 0 and "skills" in graph.nodes[node[0]]:
            skls = set(graph.nodes[node[0]]["skills"].split(","))
            for skill in skls:
                if skill not in total_skills:
                    total_skills[skill] = set()
                total_skills[skill].add(node[0])
    sum = 0
    for skill in total_skills.keys():
        sum += (np.power((len(total_skills[skill]) / len(uniq_members)), 2))
    return 1 / sum


def fitness(graph, team, task):
    pass

# Example fitness function: Sum of the values (modify as needed)
# sd = sum_distance(graph, team, task)
# tg = nx.Graph()
# for u in team:
# 	for v in team:
# 		if u[0] in nx.nodes(graph) and v[0] in nx.nodes(graph) and nx.has_path(graph, u[0], v[0]):
# 			sd += nx.dijkstra_path_length(graph, u[0], v[0])
# 			cn = u[0]
# 			for nd in nx.dijkstra_path(graph, cn, v[0]):
# 				if cn != nd:
# 					tg.add_edge(cn, nd)
# 					cn = nd


def aco(graph, task):
    import networkx as nx
    import numpy as np
    # Initialize the best solution
    best_path = None
    best_length = float('inf')
    # ACO parameters
    num_ants = 10
    num_iterations = 100
    alpha = 1.0  # Pheromone importance
    beta = 1.0  # Distance importance
    evaporation_rate = 0.5
    pheromone_deposit = 1.0

    # Initialize pheromone levels
    pheromones = np.ones((nx.number_of_nodes(graph), nx.number_of_nodes(graph)))
    distances = []
    for nd1 in list(graph.nodes):
        dst = []
        for nd2 in list(graph.nodes):
            dst.append(nx.dijkstra_path_length(graph, nd1, nd2))
        distances.append(dst)
    print("hai")

    def choose_next_expert(pheromones, distances, visited, current_expert, alpha, beta):
        probabilities = []
        for city in range(len(pheromones)):
            if city not in visited:
                pheromone_level = pheromones[current_expert][city] ** alpha
                distance = (1.0 / distances[current_expert][city]) ** beta
                probabilities.append(pheromone_level * distance)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(range(len(pheromones)), p=probabilities)

    def construct_solution(pheromones, distances, alpha, beta):
        solution = []
        visited = set()
        current_city = np.random.randint(0, len(pheromones))
        solution.append(current_city)
        visited.add(current_city)

        while len(visited) < len(pheromones):
            next_city = choose_next_expert(pheromones, distances, visited, current_city, alpha, beta)
            solution.append(next_city)
            visited.add(next_city)
            current_city = next_city
        return solution

    def path_length(path, distances):
        length = 0
        for i in range(len(path) - 1):
            length += distances[path[i]][path[i + 1]]
        length += distances[path[-1]][path[0]]  # Return to the origin
        return length

    def update_pheromones(pheromones, solutions, distances, evaporation_rate, pheromone_deposit):
        # Evaporate pheromones
        pheromones *= (1 - evaporation_rate)

        # Add new pheromones
        for solution in solutions:
            length = path_length(solution, distances)
            for i in range(len(solution) - 1):
                pheromones[solution[i]][solution[i + 1]] += pheromone_deposit / length
            pheromones[solution[-1]][solution[0]] += pheromone_deposit / length

    for iteration in range(num_iterations):
        solutions = []
        for ant in range(num_ants):
            solution = construct_solution(pheromones, distances, alpha, beta)
            solutions.append(solution)

            length = path_length(solution, distances)
            if length < best_length:
                best_length = length
                best_path = solution

        update_pheromones(pheromones, solutions, distances, evaporation_rate, pheromone_deposit)
        print(f"Iteration {iteration + 1}, Best Length: {best_length}")
    print("Best Path:", best_path)
    print("Best Length:", best_length)


def genetic_algo(graph, task, popularity_skill):
    """
    returns team of experts with minimum leader distance
    :param graph:
    :param task:
    :return Team :
    """
    import pygad
    import numpy as np
    best_team = []
    start = time.time_ns()
    num_generations = 50
    num_parents_mating = 2
    belief_space = dict()
    sol_per_pop = 10
    num_genes = len(task)
    num_elite_teams = 2
    gene_space = [[int(expert) for expert in popularity_skill[skill]] for skill in task]

    parent_selection_type = "rank"
    keep_parents = 2

    crossover_type = "single_point"

    mutation_type = "adaptive"

    def on_start(ga_instance):
        # import itertools
        initial_population = []
        for _ in range(sol_per_pop):
            sol = []
            for skill in task:
                sol.append(np.random.choice(popularity_skill[skill], 1, True)[0])
            initial_population.append(sol)
        initial_population.sort()
        ga_instance.initial_population = initial_population

    # ga_instance.initial_population = list(k for k, _ in itertools.groupby(initial_population))
    # print("Initial population has been initialized.")
    # print(ga_instance.initial_population)

    def fitness_func(ga_instance, solution, solution_idx):
        teamt = []
        teamt.append((solution[0], ''))
        for skill in task:
            for expert in popularity_skill[skill]:
                if int(expert) in solution:
                    teamt.append((expert, skill))
                    break
        return 1 / sum_distance(graph, teamt,
                                task)  # The genetic algorithm expects the fitness function to be a maximization one,

    fitness_function = fitness_func

    def on_fitness(ga_instance, population_fitness):
        print("on_fitness()")

    def on_parents(ga_instance, selected_parents):
        print("on_parents()")

    def on_crossover(ga_instance, offspring_crossover):
        print("on_crossover()")

    def on_mutation(ga_instance, offspring_mutation):
        print("on_mutation()")

    def on_generation(ga_instance):
        print("on_generation()")

    # print("Generation", ga_instance.generations_completed, end=" ")
    # import itertools
    # ga_instance.population.sort()
    # ga_instance.population = list(k for k, _ in itertools.groupby(ga_instance.population))
    # print(ga_instance.population)

    def on_stop(ga_instance, last_population_fitness):
        print("on_stop()")

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           gene_type=int,
                           mutation_num_genes=(3, 1),
                           allow_duplicate_genes=True,
                           on_start=on_start,
                           keep_elitism=1,
                           K_tournament=3
                           )

    ga_instance.run()

    if ga_instance.best_solution_generation != -1:
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_team.append((solution[0], ''))
        for skill in task:
            for expert in popularity_skill[skill]:
                if int(expert) in solution:
                    best_team.append((expert, skill))
                    break
    # print("solution", sum_distance(graph, best_team, task), end="")
    return best_team, time.time_ns() - start


def cultural(graph, task, popularity_skill):
    """
    returns team of experts with minimum leader distance
    :param graph:
    :param task:
    :return Team :
    """
    import pygad
    import numpy as np
    best_team = []
    start = time.time_ns()
    num_generations = 50
    num_parents_mating = 2
    belief_space = dict()
    sol_per_pop = 10
    num_genes = len(task)
    num_elite_teams = 2
    gene_space = [[int(expert) for expert in popularity_skill[skill]] for skill in task]
    parent_selection_type = "rank"
    keep_parents = 2

    crossover_type = "single_point"

    mutation_type = "adaptive"

    def on_start(ga_instance):
        # import itertools
        initial_population = []
        for _ in range(sol_per_pop):
            sol = []
            for skill in task:
                sol.append(np.random.choice(popularity_skill[skill], 1, True)[0])
            initial_population.append(sol)
        initial_population.sort()
        ga_instance.initial_population = initial_population

    # ga_instance.initial_population = list(k for k, _ in itertools.groupby(initial_population))
    # print("Initial population has been initialized.")
    # print(ga_instance.initial_population)

    def fitness_func(ga_instance, solution, solution_idx):
        teamt = []
        teamt.append((solution[0], ''))
        for skill in task:
            for expert in popularity_skill[skill]:
                if int(expert) in solution:
                    teamt.append((expert, skill))
                    break
        return 1 / sum_distance(graph, teamt,
                                task)  # The genetic algorithm expects the fitness function to be a maximization one,

    fitness_function = fitness_func

    def on_fitness(ga_instance, population_fitness):
        print("on_fitness()")

    def on_parents(ga_instance, selected_parents):
        print("on_parents()")

    def on_crossover(ga_instance, offspring_crossover):
        print("on_crossover()")

    def on_mutation(ga_instance, offspring_mutation):
        print("on_mutation()")

    def on_generation(ga_instance):
        # print("on_generation()")
        elite_teams = []
        if len(ga_instance.population) > num_elite_teams:
            elite_teams = [ga_instance.population[i] for i in range(num_elite_teams)]
        else:
            elite_teams = [ga_instance.population[i] for i in range(len(ga_instance.population))]
        for etm in elite_teams:
            for mmbr in etm:
                for skill in task:
                    if skill not in belief_space:
                        belief_space[skill] = list()
                        belief_space[skill].append(mmbr)
                        break
        ga_instance.gene_space = [[int(expert) for expert in belief_space[skill]] for skill in task]

    # print(ga_instance.population)

    def on_stop(ga_instance, last_population_fitness):
        print("on_stop()")

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           gene_type=int,
                           mutation_num_genes=(3, 1),
                           allow_duplicate_genes=True,
                           on_start=on_start,
                           on_generation=on_generation,
                           keep_elitism=1,
                           K_tournament=3
                           )

    ga_instance.run()

    if ga_instance.best_solution_generation != -1:
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_team.append((solution[0], ''))
        for skill in task:
            for expert in popularity_skill[skill]:
                if int(expert) in solution:
                    best_team.append((expert, skill))
                    break
    # print("solution", sum_distance(graph, best_team, task), end="")
    return best_team, time.time_ns() - start


# def update_belief_space(self):
# 	self.belief_space = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.belief_space_size]
#
#
# def influence(self):
# 	for individual in self.population:
# 		if random.random() < self.mutation_rate:
# 			# Influence from belief space
# 			influence_source = random.choice(self.belief_space)
# 			individual.team = influence_source.team

def minLD(graph, task, popularity_skill):
    """
    returns team of experts with minimum leader distance
    :param graph:
    :param task:
    :return Team :
    """
    import networkx as nx
    import sys
    start = time.time_ns()
    best_team = []
    best_ld = sys.maxsize
    for leader in graph.nodes:
        team = []
        team.append((leader, ""))
        for skill in task:
            close_distance = sys.maxsize
            closest_expert = ""
            for expert in popularity_skill[skill]:
                if nx.has_path(graph, leader, expert):
                    distance = nx.dijkstra_path_length(graph, leader, expert, weight="weight")
                    if close_distance > distance:
                        close_distance = distance
                        closest_expert = expert
            team.append((closest_expert, skill))
        if len(team) >= len(task):
            leaderdistance = leader_distance(graph, team)
            if leaderdistance < best_ld:
                best_ld = leaderdistance
                best_team = team
    # print(leader, leaderdistance, team)
    return best_team, time.time_ns() - start


def minSD(graph, task, popularity_skill):
    """
    returns team of experts with minimum leader distance
    :param graph:
    :param task:
    :return Team :
    """
    import networkx as nx
    import sys
    start = time.time_ns()
    best_team = []
    best_sd = sys.maxsize
    for skill1 in task:
        for leader in popularity_skill[skill1]:
            team = []
            dtask = task.copy()
            team.append((leader, ""))
            team.append((leader, skill1))
            dtask.remove(skill1)
            for skill2 in dtask:
                close_distance = sys.maxsize
                closest_expert = ""
                for expert in popularity_skill[skill2]:
                    if nx.has_path(graph, leader, expert):
                        distance = nx.dijkstra_path_length(graph, leader, expert, weight="weight")
                        if close_distance > distance:
                            close_distance = distance
                            closest_expert = expert
                team.append((closest_expert, skill2))
            if len(team) >= len(task):
                sumdistance = sum_distance(graph, team, task)
                if sumdistance < best_sd:
                    best_sd = sumdistance
                    best_team = team
    # print(leader, sumdistance, team)
    return best_team, time.time_ns() - start


def TPLRandom(graph, task, popularity_skill, hops, lmbda):  # twice of average degree
    """
    return community based team formation using closest expert.
    :param graph:
    :param task:
    :return:
    """
    import random
    start = time.time_ns()
    avg_degree = (2 * graph.number_of_edges()) / float(graph.number_of_nodes())
    hc = [node for node in list(graph.nodes) if graph.degree[node] > (lmbda * avg_degree)]
    hcws = []  # High collaborating nodes with skills
    for nd in hc:
        if (len(graph.nodes[nd]) > 0 and len(set(graph.nodes[nd]["skills"].split(",")).intersection(set(task))) > 0):
            hcws.append(nd)
    best_team = []
    best_random = []
    best_ld = sys.maxsize
    for node in hcws:
        random_experts = []
        team = []
        leader = node
        dtask = task.copy()
        team.append((leader, ""))
        covered = list(set(task).intersection(set(graph.nodes[leader]["skills"].split(","))))
        for skill1 in covered:
            team.append((leader, skill1))
            dtask.remove(skill1)
        hop_nodes = within_k_nbrs(graph, leader, hops)
        neighborhood = []
        for inode in hop_nodes:
            if len(graph.nodes[inode]) > 0:
                skills = set(graph.nodes[inode]["skills"].split(",")).intersection(dtask)
                if len(skills) > 0:
                    neighborhood.append([inode, skills])
        neighborhood.sort(key=lambda elem: (-len(elem[1])))  # sort neighborhood max skills
        flag = True
        while flag:
            if not dtask:  # empty list is false
                flag = False
            elif not neighborhood:
                flag = False
            else:
                expert = neighborhood.pop(0)
                for skl in list((expert[1])):
                    if skl in dtask:
                        team.append((expert[0], skl))
                        dtask.remove(skl)
                neighborhood = neighborhood[1:]
        tsk_lst = list(dtask)
        while len(tsk_lst) > 0:
            skill = random.choice(tsk_lst)
            random_expert = random.choice(popularity_skill[skill])
            team.append((random_expert, skill))
            random_experts.append((random_expert, skill))
            tsk_lst.remove(skill)
        if len(team) >= len(task):
            leaderdistance = leader_distance(graph, team)
            if leaderdistance < best_ld:
                best_ld = leaderdistance
                best_team = team
                best_random = random_experts
    # print(leader, leaderdistance, team)
    return best_team, best_random, time.time_ns() - start


def TPLClosest(graph, task, popularity_skill, hops, lmbda):  # twice of average degree
    """
    return community based team formation using closest expert.
    :param graph:
    :param task:
    :return:
    """
    import random
    start = time.time_ns()
    avg_degree = (2 * graph.number_of_edges()) / float(graph.number_of_nodes())
    hc = [node for node in list(graph.nodes) if graph.degree[node] > (lmbda * avg_degree)]
    hcws = []  # High collaborating nodes with skills
    for nd in hc:
        if (len(graph.nodes[nd]) > 0 and len(set(graph.nodes[nd]["skills"].split(",")).intersection(set(task))) > 0):
            hcws.append(nd)
    best_team = []
    best_random = []
    best_ld = sys.maxsize
    for node in hcws:
        random_experts = []
        team = []
        leader = node
        dtask = task.copy()
        team.append((leader, ""))
        covered = list(set(task).intersection(set(graph.nodes[leader]["skills"].split(","))))
        for skill1 in covered:
            team.append((leader, skill1))
            dtask.remove(skill1)
        hop_nodes = within_k_nbrs(graph, leader, hops)
        neighborhood = []
        for inode in hop_nodes:
            if len(graph.nodes[inode]) > 0:
                skills = set(graph.nodes[inode]["skills"].split(",")).intersection(dtask)
                if len(skills) > 0:
                    neighborhood.append([inode, skills])
        neighborhood.sort(key=lambda elem: (-len(elem[1])))  # sort neighborhood max skills
        flag = True
        while flag:
            if not dtask:  # empty list is false
                flag = False
            elif not neighborhood:
                flag = False
            else:
                expert = neighborhood.pop(0)
                for skl in list((expert[1])):
                    if skl in dtask:
                        team.append((expert[0], skl))
                        dtask.remove(skl)
                neighborhood = neighborhood[1:]
        tsk_lst = list(dtask)
        while len(tsk_lst) > 0:
            skill = random.choice(tsk_lst)
            close_distance = sys.maxsize
            closest_expert = ""
            for expert in popularity_skill[skill]:
                if nx.has_path(graph, leader, expert):
                    distance = nx.dijkstra_path_length(graph, leader, expert, weight="weight")
                    if close_distance > distance:
                        close_distance = distance
                        closest_expert = expert
            team.append((closest_expert, skill))
            random_experts.append((closest_expert, skill))
            tsk_lst.remove(skill)
        if len(team) >= len(task):
            leaderdistance = leader_distance(graph, team)
            if leaderdistance < best_ld:
                best_ld = leaderdistance
                best_team = team
                best_random = random_experts
    # print(leader, leaderdistance, team)
    return best_team, best_random, time.time_ns() - start


# Ref: https://stackoverflow.com/questions/18393842/k-th-order-neighbors-in-graph-python-networkx
def within_k_nbrs(grap, start, k):
    nbrs = {start}
    for _ in range(k):
        nbrs = set((nbr for n in nbrs for nbr in grap[n]))
    return nbrs


def rarestfirst(graph, task, popularity_skill):
    """
    returns team of experts with minimum diameter distance
    :param graph:
    :param task:
    :return tuple(set, dictionary, string):
    """
    import sys
    start = time.time_ns()
    rareskill = ""
    best_dd = sys.maxsize
    skill_support = 0
    best_team = []
    for skill in task:
        if len(popularity_skill[skill]) > skill_support:
            skill_support = len(popularity_skill[skill])
            rareskill = skill
    for leader in popularity_skill[rareskill]:
        team = []
        team.append((leader, ""))
        team.append((leader, rareskill))
        for skill in task:
            if skill != rareskill:
                close_distance = sys.maxsize
                closest_expert = ""
                for expert in popularity_skill[skill]:
                    if nx.has_path(graph, leader, expert):
                        distance = nx.dijkstra_path_length(graph, leader, expert, weight="weight")
                        if close_distance > distance:
                            close_distance = distance
                            closest_expert = expert
                team.append((closest_expert, skill))
        if len(team) >= len(task):
            diameterdistance = diameter_distance(graph, team)
            if diameterdistance < best_dd:
                best_dd = diameterdistance
                best_team = team
    # print(leader, diameterdistance, team)
    return best_team, time.time_ns() - start


def diameter_distance(graph, team) -> float:
    team_nodes = set()
    leader = team[0][0]
    team_nodes.add(leader)
    uniq_members = set([mmbr[0] for mmbr in team])
    for expert in uniq_members:
        for node in nx.dijkstra_path(graph, leader, expert, weight="weight"):
            team_nodes.add(node)
    team_graph = nx.subgraph(graph, team_nodes).copy()
    return nx.diameter(team_graph, weight="weight")


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


def leader_distance(graph, team) -> float:
    """
    return leader distance of team i.e. (leader, team_member) pairs
    :param graph:
    :return:
    """
    import networkx as nx
    leaderdistance = 0
    uniq_members = set([mmbr[0] for mmbr in team])
    if len(team) < 2:
        return 0
    else:
        leader = team[0][0]
        for member in uniq_members:
            if member != leader and nx.has_path(graph, leader, member):
                leaderdistance += nx.dijkstra_path_length(graph, leader, member, weight="weight")
    return round(leaderdistance, 3)


if __name__ == '__main__':
    tasks = []
    # Toy Example
    graph = nx.read_gml("/home/ramesh/dblp/input/icdt.gml")
    with open("/home/ramesh/dblp/input/icdt_tasks.txt") as file:
        for line in file:
            task = [x for x in line.strip("\n").split("\t") if x]
            tasks.append(task)
    popularity_skill = dict()  # experts_for_skill i.e. skill:list of experts
    for node in graph.nodes:
        if "skills" in graph.nodes[node]:
            for skill in graph.nodes[node]["skills"].split(","):
                if skill in popularity_skill:
                    popularity_skill[skill].append(node)
                else:
                    popularity_skill[skill] = list()
                    popularity_skill[skill].append(node)
    for skill in popularity_skill:
        popularity_skill[skill] = list(set(popularity_skill[skill]))
    for task in tasks:
        print(blenddtfp(graph, task, popularity_skill))

# database_name = "db"
# network = nx.read_gml("/home/ramesh/diversity/input/" + database_name + ".gml")
# i = 1
# with open("/home/ramesh/diversity/input/" + database_name + "_tasks.txt") as file:
# 	for line in file:
# 		task = [x for x in line.strip("\n").split("\t") if x]
# 		tasks.append(task)
# aco(network, task)
# open("/home/ramesh/diversity/output/" + database_name + "_blend_dtfp.txt", "w").close()
# t, g, d, sd, gs, shn = 0, 0, 0, 0, 0, 0
# gt, gg, gd, gsd, ggs, gshn = 0, 0, 0, 0, 0, 0
# for task in tasks:
# 	t, g, d, sd, gs, shn = blenddtfp(network, task)
# 	gt += t
# 	gg += g
# 	gd += d
# 	gsd += sd
# 	ggs += gs
# 	gshn += shn
# 	if i % 10 == 0:
# 		with open("/home/ramesh/diversity/output/" + database_name + "_blend_dtfp.txt","a") as file:
# 			file.write(str(gt / 10)+", "+str(gg / 10)+", "+str(gd / 10)+", "+str(gsd / 10)+", "+
# 			           str(ggs / 10)+", "+str(gshn / 10)+"\n")
# 		t, g, d, sd, gs, shn = 0, 0, 0, 0, 0, 0
# 		gt, gg, gd, gsd, ggs, gshn = 0, 0, 0, 0, 0, 0
# 	i+=1