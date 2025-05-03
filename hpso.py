import random
import numpy as np
import networkx as nx

# Define Graph (using networkx)
graph = nx.read_gml("/home/ramesh/dblp/input/icdt.gml")

# Objective 1: Maximize node properties (sum of node weights)
def gamma_diversity(graph, team, task):
    gama = set()
    for nd in team:
        gama.update(set(graph.nodes[nd[0]]["skills"].split(",")))
    return len(gama)


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

def unique_teams(listofteams):
  unique_teams = []
  for team in listofteams:
      if team not in unique_teams:
          unique_teams.append(team)
  return unique_teams

import numpy as np
import random

# Number of individuals and teams
N = nx.number_of_nodes(graph)  # Number of individuals
T = 1  # Number of teams
iterations = 100  # Number of iterations for PSO
population_size = 10  # Number of particles

# Hyperparameters for PSO
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive component
c2 = 1.5  # Social component

# Fitness function: minimize communication cost and maximize diversity
def calculate_fitness(position):
    teams = {i: [] for i in range(1, T+1)}
    for idx, team in enumerate(position):
        teams[team].append(idx)

    # Calculate communication cost
    comm_cost = sum_distance(graph, team, task)

    # Calculate diversity
    diversity_score = gamma_diversity(graph, team, task)

    # We want to minimize comm_cost and maximize diversity
    # Fitness = w1 * comm_cost - w2 * diversity_score
    # Let's assume w1 = 1 and w2 = 1 for simplicity
    return diversity_score / comm_cost

# Initialize particles (positions and velocities)
def initialize_particles(population_size, N):
    particles = []
    velocities = []
    for _ in range(population_size):
        sol = []
        for skill in task:
            sol.append([np.random.choice(list(popularity_skill[skill]), 1, True)[0], skill])
        velocity = [random.uniform(-1, 1) for _ in range(N)]
        particles.append(sol)
        velocities.append(velocity)
    return particles, velocities

# Update the velocity and position of each particle
def update_particle(particle, velocity, pbest, gbest):
    new_velocity = []
    new_position = []
    for i in range(len(particle)):
        r1 = random.random()
        r2 = random.random()
        new_v = (w * velocity[i] +
                 c1 * r1 * (pbest[i] - particle[i]) +
                 c2 * r2 * (gbest[i] - particle[i]))
        new_velocity.append(new_v)

        # Update position using the new velocity
        new_p = particle[i] + new_v
        # Ensure the position is within the team bounds (discrete values: 1 to T)
        new_p = int(round(new_p))
        if new_p < 1:
            new_p = 1
        if new_p > T:
            new_p = T
        new_position.append(new_p)

    return new_position, new_velocity

# PSO algorithm
def pso_team_formation():
    # Initialize particles and velocities
    particles, velocities = initialize_particles(population_size, N)

    # Initialize personal best (pbest) and global best (gbest)
    pbest = particles[:]
    pbest_fitness = [calculate_fitness(p) for p in particles]
    gbest = particles[np.argmin(pbest_fitness)]
    gbest_fitness = min(pbest_fitness)

    # Start iterations
    for iteration in range(iterations):
        for i in range(population_size):
            # Update particle velocity and position
            particles[i], velocities[i] = update_particle(particles[i], velocities[i], pbest[i], gbest)

            # Calculate the fitness of the new position
            fitness = calculate_fitness(particles[i])

            # Update personal best if necessary
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i]
                pbest_fitness[i] = fitness

            # Update global best if necessary
            if fitness < gbest_fitness:
                gbest = particles[i]
                gbest_fitness = fitness

        # Print the best fitness found so far
        print(f"Iteration {iteration+1}/{iterations}, Best Fitness: {gbest_fitness}")

    return gbest, gbest_fitness

import networkx as nx

# networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
#             "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
networks = ["icdt"]
for network in networks:
    print(network)
    tasks = []
    graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
    with open("/home/ramesh/dblp/input/" + network + "_tasks.txt") as file:
        for line in file:
            task = [x for x in line.strip("\n").split("\t") if x]
            tasks.append(task)
    popularity_skill = dict()  # experts_for_skill i.e. skill:list of experts
    for node in graph.nodes:
        if "skills" in graph.nodes[node]:
            for skill in graph.nodes[node]["skills"].split(","):
                if skill in popularity_skill:
                    popularity_skill[skill].add(node)
                else:
                    popularity_skill[skill] = set()
                    popularity_skill[skill].add(node)
    open("/home/ramesh/dblp/output/" + network + "_pso_teams.txt", "w").close()
    for task in tasks:
        best_team_configuration, best_fitness = pso_team_formation()   # Run PSO for team formation
        print(f"Best Team Configuration: {best_team_configuration}")
        print(f"Best Fitness (Lower is Better): {best_fitness}")
