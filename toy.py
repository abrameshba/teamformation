import networkx as nx
import matplotlib.pyplot as plt

def construct_toy():
	graph = nx.Graph()
	skills = {1: "java", 2: "javascript", 3: "html", 4: "db", 5: "bash", 6: "css"}
	nodes = [(1, {"name": "A", "skills": "1,2"}), (2, {"name": "B", "skills": "4"}), (3, {"name": "C", "skills": "3"}),
	         (4, {"name": "D", "skills": "1,2"}), (5, {"name": "E", "skills": "4"}), (6, {"name": "F", "skills": "3"}),
	         (7, {"name": "G", "skills": "5,6"})]
	graph.add_nodes_from(nodes)
	edges = [(1,3,{"weight":6}), (2,3,{"weight":2}),(1,2,{"weight":7}),(1,4,{"weight":5}),
	         (1,7,{"weight":4}), (4,5,{"weight":3}),(4,6,{"weight":4}),(5,6,{"weight":6}),(4,7,{"weight":3})]
	graph.add_edges_from(edges)
	nx.draw(graph, with_labels=True)
	plt.show()
	nx.write_gml(graph, "/home/ramesh/diversity/input/toy.gml")
construct_toy()