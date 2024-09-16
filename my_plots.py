# Code to convert from Gomez followed network structure to custom
# import networkx as nx
# bibs = ["nature", "physica", "science"]
# for network in bibs:
#     graph = nx.read_gml("/home/ramesh/diversity/input/" + network + ".gml")
#     dicts = dict()
#     for node in graph.nodes:
#         skild = graph.nodes[node]
#         skills = ",".join([key.split("_")[1] for key in skild if skild[key]!='0'])
#         dicts[node]={"skills":skills}
#     mygraph = nx.Graph()
#     mygraph.add_nodes_from([nd for nd in graph.nodes])
#     mygraph.add_edges_from([edg for edg in graph.edges])
#     nx.set_node_attributes(mygraph, dicts)
#     nx.write_gml(mygraph, "/home/ramesh/diversity/input/" + network + "-.gml")

import pandas as pd

# networks = ["nature", "physica", "science"]
networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
			"focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
algorithms = ["minLD", "minSD", "rarestfirst", "TPLR11", "TPLR22"]
tasksize = [i for i in range(4, 11)]
import matplotlib.pyplot as plt
import seaborn as sns

color = ["green", "white", "red"]
for network in networks:
	dataframe = pd.DataFrame(
		columns=["algorithm", "task_size", "team_size", "diameter", "leader_distance", "sum_distance"])
	for algo in algorithms:
		with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt") as file:
			lc = 0  # line count
			i = 0
			for line in file:
				lst = [algo]
				lst.extend(itm for itm in line.strip("\n").split("\t"))
				dataframe.loc[len(dataframe.index)] = lst
				lc += 1
				if lc % 5 == 0:
					i += 1
	# sns.boxplot(data=dataframe, x="task_size", y="leader_distance", hue="algorithm")
	sns.lineplot(data=dataframe, x="task_size", y="sum_distance", hue="algorithm")
	plt.gca().invert_yaxis()
	plt.yscale("log")
	# sns.scatterplot(data=dataframe, x="task_size", y="processing_time", hue="algorithm")
	plt.savefig("/home/ramesh/dblp/output/eps/" + network + "_sum_distance.eps", format="eps")
	plt.show()

# import numpy as np
# import random
# import matplotlib.pyplot as plt
#
#
# class NSGA2:
#     def __init__(self, graph, pop_size, num_generations, crossover_prob, mutation_prob, task):
#         self.init_values = list(graph.nodes)
#         self.graph = graph
#         self.pop_size = pop_size
#         self.task = task
#         self.num_generations = num_generations
#         self.crossover_prob = crossover_prob
#         self.mutation_prob = mutation_prob
#         self.num_genes = len(task)
#         self.population = self.initialize_population()
#         self.best_fitness_history = []
#
#
#     def initialize_population(self):
#         ppl = []
#         for node in list(graph.nodes):
#             if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
#                 skls = set(graph.nodes[node]["skills"].split(","))
#                 if len(skls.intersection(set(self.task))) > 0:
#                     ppl.append(node)
#         return [random.sample(ppl, len(self.task)) for _ in range(self.pop_size)]
#
#     def evaluate_population(self, population):
#         rmd = []
#         for ppl in population:
#             cnt = 0
#             for i in range(len(self.task)):
#                 if len(graph.nodes[ppl[i]]) > 0 and "skills" in graph.nodes[ppl[i]]:
#                     skls = graph.nodes[ppl[i]]["skills"].split(",")
#                     if len([i for i in task if i in skls]) > 0:
#                         cnt+=1
#             if cnt < len(self.task):
#                 rmd.append(ppl)
#         for ppl in rmd:
#             population.remove(ppl)
#         ft = []
#         if len(population)>0:
#             for ppl in population:
#                 ft.append(self.fitness(ppl))
#         return ft
#
#
#     def fitness(self, individual):
#         # Example fitness function: Sum of the values (modify as needed)
#         return np.sum(individual)
#
#     def selection(self, population, fitness):
#         selected_indices = []
#         for _ in range(self.pop_size):
#             i, j = np.random.choice(self.pop_size, 2, replace=False)
#             if fitness[i] < fitness[j]:
#                 selected_indices.append(i)
#             else:
#                 selected_indices.append(j)
#         return [population[i] for i in selected_indices]
#
#     def crossover(self, parent1, parent2):
#         if np.random.rand() < self.crossover_prob:
#             point = np.random.randint(1, self.num_genes - 1)
#             child1 = parent1[:point] + parent2[point:]
#             child2 = parent2[:point] + parent1[point:]
#             return child1, child2
#         else:
#             return parent1, parent2
#
#     def mutate(self, individual):
#         for i in range(self.num_genes):
#             if np.random.rand() < self.mutation_prob:
#                 swap_idx = np.random.randint(0, self.num_genes)
#                 individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
#         return individual
#
#     def run(self):
#         best_solution = None
#         best_fitness = np.inf
#
#         for generation in range(self.num_generations):
#             fitness = self.evaluate_population(self.population)
#             if len(fitness) > 0:
#                 best_idx = np.argmin(fitness)
#                 if fitness[best_idx] < best_fitness:
#                     best_fitness = fitness[best_idx]
#                     best_solution = self.population[best_idx]
#
#                 self.best_fitness_history.append(best_fitness)
#
#                 selected_population = self.selection(self.population, fitness)
#                 offspring_population = []
#
#                 for i in range(0, self.pop_size, 2):
#                     parent1, parent2 = selected_population[i], selected_population[i + 1]
#                     child1, child2 = self.crossover(parent1, parent2)
#                     offspring_population.append(self.mutate(child1))
#                     offspring_population.append(self.mutate(child2))
#
#                 self.population = offspring_population
#
#         return best_solution, best_fitness, self.best_fitness_history
#
#
# # Example usage
# import networkx as nx
# database_name = "icdt"
# graph = nx.read_gml("/home/ramesh/diversity/input/" + database_name + ".gml")
# tasks = []
# with open("/home/ramesh/diversity/input/" + database_name + "_tasks.txt") as file:
#     for line in file:
#         task = [x for x in line.strip("\n").split("\t") if x]
#         tasks.append(task)
# init_values = list(graph.nodes)
# pop_size = 100
# num_generations = 100
# crossover_prob = 0.9
# mutation_prob = 0.1
# for task in tasks:
#     nsga2 = NSGA2(graph, pop_size, num_generations, crossover_prob, mutation_prob, tasks)
#     best_solution, best_fitness, best_fitness_history = nsga2.run()
#
# print(f"Best Solution: {best_solution}")
# print(f"Best Fitness: {best_fitness}")
#
# # Plot the fitness history
# plt.plot(best_fitness_history)
# plt.xlabel('Generation')
# plt.ylabel('Best Fitness')
# plt.title('Best Fitness Over Generations')
# plt.show()
#
