def construct_graph():
    """
    This method is used to construct graph from PPI_edges.txt file containing edges information With weights
    ensg.A	ensg.B	score
    ENSG00000081189	ENSG00000090861	0.04
    and PPI_nodes_details.txt file containing nodes(proteins) attributes:
    ensg	gene_name	biotype     CA1	    CA2	    CA3	    CA4     DG
    ENSG00000000419 DPM1 protein_coding 1.06982389994692 1.06296794238889 1.03941285747258 1.02444076378951 1.09348202829185
    :return: None
    """
    import networkx as nx
    graph = nx.MultiDiGraph()
    with open("/home/ramesh/HENA/PPI_nodes_details.txt") as nodes_file:
        for line in nodes_file:
            words = line.strip("\n").split(" ")
            # G.add_node(words[0], gene_name=words[1], biotype=words[2])
            graph.add_nodes_from([(words[0], {"gene_name": words[1], "biotype": words[2], "CA1": words[3], "CA2": words[4],
                                       "CA3": words[5], "CA4": words[6], "DG":words[7]})])
        del line, words
    with open("/home/ramesh/HENA/PPI_edges.txt") as edges_file:
        for line in edges_file:
            words = line.strip("\n").split(" ")
            graph.add_edge(words[0], words[1], weight=words[2])
        del line, words
    del nodes_file, edges_file
    nx.write_gml(G=graph, path="/home/ramesh/HENA/PPI.gml")
    for u in list(graph.nodes):
        for v in list(graph.nodes):
            if u == v and graph.has_edge(u,v):
                graph.remove_edge(u,v)
            else:
                if graph.get_edge_data(u, v) is not None and len(graph.get_edge_data(u, v)) > 1:
                    print(u, v, len(graph.get_edge_data(u, v)), graph.get_edge_data(u, v))
    # Gcc = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    # print(len(list(nx.selfloop_edges(graph))))
    # graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    # nx.write_gml(G=graph, path="/home/ramesh/HENA/PPI_no_loops.gml")
    # lcc = graph.subgraph(Gcc[0])
    # nx.write_gml(G=lcc, path="/home/ramesh/HENA/PPI_lcc.gml")
    # nx.draw(graph)
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == "__main__":
    construct_graph()
