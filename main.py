# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from random import sample

import networkx as nx

import algorithms


def generate_tasks():
    # networks = ["nature", "physica", "science"]
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        uskills = set()
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(","))
                uskills.update(skls)
        uskls = sorted(list(uskills))
        with open("/home/ramesh/dblp/input/" + network + "_tasks.txt", "w") as file:
            for ts in range(4, 21):
                for _ in range(1, 6):
                    task = sample(uskls, ts)
                    file.write("\t".join(task) + "\n")


def check_toy_team():
    networks = ["icdt"]
    for network in networks:
        print(network)
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        tasks = []
        with open("/home/ramesh/dblp/input/" + network + "_tasks.txt") as file:
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
        i = 1
        for task in tasks:
            teamld, timeld = algorithms.minLD(graph, task, popularity_skill)
            uniq_ld = set([mmbr[0] for mmbr in teamld])
            teamsd, timesd = algorithms.minSD(graph, task, popularity_skill)
            uniq_sd = set([mmbr[0] for mmbr in teamsd])
            teamrf, timerf = algorithms.rarestfirst(graph, task, popularity_skill)
            uniq_rf = set([mmbr[0] for mmbr in teamrf])
            all = uniq_rf.union(uniq_sd, uniq_ld)
            print(str(i), all, task)
            print("ld %s %s %s %s" % (
                algorithms.leader_distance(graph, teamld), algorithms.sum_distance(graph, teamsd, task),
                algorithms.diameter_distance(graph, teamld), uniq_ld))
            print("sd %s %s %s %s" % (
                algorithms.leader_distance(graph, teamsd), algorithms.sum_distance(graph, teamsd, task),
                algorithms.diameter_distance(graph, teamsd), uniq_sd))
            print("rf %s %s %s %s" % (
                algorithms.leader_distance(graph, teamrf), algorithms.sum_distance(graph, teamrf, task),
                algorithms.diameter_distance(graph, teamrf), uniq_rf))
            i += 1


def experiment():
    # networks = ["nature", "physica", "science"]
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        print(network)
        open("/home/ramesh/dblp/output/" + network + "_blend_teams.txt", "w").close()
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        tasks = []
        with open("/home/ramesh/dblp/input/" + network + "_tasks.txt") as file:
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
        for task in tasks:
            # with open("/home/ramesh/dblp/output/" + network + "_TPLR11_teams.txt", "a") as file:
            # 	team, rndm, time = algorithms.TPLRandom(graph, task, popularity_skill, 1, 1)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join('%s,%s' % x for x in rndm)+ "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_TPLR22_teams.txt", "a") as file:
            # 	team, rndm,  time = algorithms.TPLRandom(graph, task, popularity_skill, 2, 2)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join('%s,%s' % x for x in rndm)+ "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_TPLC11_teams.txt", "a") as file:
            # 	team, rndm,  time = algorithms.TPLClosest(graph, task, popularity_skill, 1, 1)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join('%s,%s' % x for x in rndm)+ "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_TPLC22_teams.txt", "a") as file:
            # 	team, rndm,  time = algorithms.TPLClosest(graph, task, popularity_skill, 2, 2)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join('%s,%s' % x for x in rndm)+ "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_minLD_teams.txt", "a") as file:
            # 	team, time = algorithms.minLD(graph, task, popularity_skill)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_minSD_teams.txt", "a") as file:
            # 	team, time = algorithms.minSD(graph, task, popularity_skill)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_rarestfirst_teams.txt", "a") as file:
            # 	team, time = algorithms.rarestfirst(graph, task, popularity_skill)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_genetic_teams.txt", "a") as file:
            # 	team, time = algorithms.genetic_algo(graph, task, popularity_skill)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/" + network + "_cultural_teams.txt", "a") as file:
            # 	team, time = algorithms.cultural(graph, task, popularity_skill)
            # 	file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            with open("/home/ramesh/dblp/output/" + network + "_blend_teams.txt", "a") as file:
                team, time = algorithms.blenddtfp(graph, task, popularity_skill)
                file.write(str(time) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")


def make_results():
    networks = ["icdt"]
    # networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
    #             "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    ts = [i for i in range(4, 21)]
    for network in networks:
        for algo in ["rarestfirst", "blend"]:
            graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt", "w").close()
            with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_teams.txt") as file:
                lc = 0
                i = 0
                for line in file:
                    lc += 1
                    result = [x for x in line.strip("\n").split("\t") if x]
                    team = [(result[1], "")]
                    for k in range(len(result)):
                        if k > 1 and k % 2 == 0:
                            team.append(((result[k], result[k + 1])))
                    task = [result[k] for k in range(len(result)) if k > 2 and k % 2 != 0]
                    ld = algorithms.leader_distance(graph, team)
                    sd = algorithms.sum_distance(graph, team, task)
                    dd = algorithms.diameter_distance(graph, team)
                    tms = len(set([mmbr[0] for mmbr in team]))
                    gama = algorithms.gamma_diversity(graph, team, task)
                    shn = algorithms.shannon_diversity(graph, team, task)
                    igsn = algorithms.inverse_gini_simpson_diversisty(graph, team, task)
                    with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt", "a") as file1:
                        file1.write(str(ts[i]) + "\t" + result[0] + "\t" + str(tms) + "\t" + str(dd) + "\t" + str(ld) +
                                    "\t" + str(sd) + "\t" + str(gama) + "\t" + str(shn) + "\t" + str(igsn) + "\n")
                    if lc % 5 == 0:
                        i += 1


def analysis():
    networks = ["icdt"]
    # networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
    #             "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    ts = [i for i in range(4, 21)]
    for network in networks:
        for algo in ["rarestfirst", "blend"]:
            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_analysis.txt", "w").close()
            with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt", "r") as file:
                lc = 0
                i = 0
                fsum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                for line in file:
                    lc += 1
                    result = [x for x in line.strip("\n").split("\t") if x]
                    tsum = [fsum[i] + float(result[i]) for i in range(len(result))]
                    fsum = tsum
                    if lc % 5 == 0:
                        i += 1
                        open("/home/ramesh/dblp/output/" + network + "_" + algo + "_analysis.txt", "a").write(
                            "\t".join([str(fsum[i] / 5) for i in range(len(result))]) + "\n")
                        tsum = fsum = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def network_details():
    # networks = ["nature", "physica", "science"]
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        uskills = set()
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        avg_degree = (2 * nx.number_of_edges(graph)) / nx.number_of_nodes(graph)
        h1 = set()
        h2 = set()
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(","))
                uskills.update(skls)
            if graph.degree[node] > avg_degree:
                h1.add(node)
            if graph.degree[node] > (2 * avg_degree):
                h2.add(node)
        print(network, nx.number_of_nodes(graph), nx.number_of_edges(graph), len(uskills), nx.diameter(graph),
              round(avg_degree, 2), len(h1), len(h2), round(len(h1) / nx.number_of_nodes(graph), 2),
              round(len(h2) / nx.number_of_nodes(graph), 2))


if __name__ == '__main__':
    # generate_tasks()
    # experiment()
    make_results()
    analysis()
    # network_details()
