# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict
from random import sample

import networkx as nx
import numpy as np
import pandas
from scipy.special.cython_special import huber
from sphinx.addnodes import index
from sympy import ceiling, floor

import algorithms


def generate_tasks():
    # networks = ["nature", "physica", "science"]
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        uskills = set()  # Universal skill set
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(", "))
                uskills.update(skls)
        uskls = sorted(list(uskills))
        with open("/home/ramesh/dblp/input/" + network + "_tasks.txt", "w") as file:
            for ts in range(4, 21):
                for _ in range(1, 6):
                    task = np.random.choice(uskls, size=ts, replace=False).tolist()
                    file.write("\t".join(task) + "\n")


def generate_rait_tasks():
    # networks = ["nature", "physica", "science"]
    import json
    networks = ["colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        uskills = set()  # Universal skill set
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(", "))
                uskills.update(skls)
        uskills = {skl.strip() for skl in uskills if len(skl) > 0}
        total_experts = 0
        ps = set()
        rs = set()
        check_skl = set()
        with open("/home/ramesh/dblp/output/" + network + "_skill_popularity.txt", "r") as file:
            for line in file:
                words = line.strip("\n").split("\t")
                check_skl.add(words[0])
                total_experts += len(words[1].split(","))
        avg_popularity = total_experts / len(uskills)
        with open("/home/ramesh/dblp/output/" + network + "_skill_popularity.txt", "r") as file:
            for line in file:
                words = line.strip("\n").split("\t")
                if len(words[1].split(",")) >= avg_popularity:
                    ps.add(words[0])
                else:
                    rs.add(words[0])
        # print(network.upper(),len(ps),round(avg_popularity,2), len(rs),len(check_skl),len(uskills))
        # for j in range(1, 101):
        #     print(j,network)
        #     with open("/home/ramesh/dblp/input/rait/" + network + "_tasks.txt", "w") as file:
        #         psd = defaultdict()
        #         for ts in range(4, 21):
        #             psc = defaultdict()
        #             for i in range(0,ts+1):
        #                 psc[i]=0
        #             for _ in range(1, 101):
        #                 task = sample(sorted(uskills), ts)
        #                 file.write("\t".join(task) + "\n")
        #                 nps = 0
        #                 for skl in task:
        #                     if skl in ps:
        #                         nps+=1
        #                 psc[nps]+=1
        #             psd[ts] = psc
        #         with open("/home/ramesh/dblp/input/rait/" + network +"_tasks_summary_"+str(j)+".txt", "a") as f:
        #             json.dump(psd, f, indent=4)
        with open("/home/ramesh/dblp/input/rait/" + network + "ar_tasks.txt", "w") as file:
            for ts in range(4, 21):
                for _ in range(1, 6):
                    task = sample(sorted(rs), ts)
                    file.write("\t".join(task) + "\n")
        with open("/home/ramesh/dblp/input/rait/" + network + "ap_tasks.txt", "w") as file:
            for ts in range(4, 21):
                for _ in range(1, 6):
                    task = sample(sorted(ps), ts)
                    file.write("\t".join(task) + "\n")
        with open("/home/ramesh/dblp/input/rait/" + network + "2r8p_tasks.txt", "w") as file:
            for ts in range(4, 21):
                for _ in range(1, 6):
                    task = sample(sorted(rs), ceiling(ts * .2))
                    task.extend(sample(sorted(ps), floor(ts * .8)))
                    file.write("\t".join(task) + "\n")
        with open("/home/ramesh/dblp/input/rait/" + network + "8r2p_tasks.txt", "w") as file:
            for ts in range(4, 21):
                for _ in range(1, 6):
                    task = sample(sorted(rs), ceiling(ts * .8))
                    task.extend(sample(sorted(ps), floor(ts * .2)))
                    file.write("\t".join(task) + "\n")


def rait_summary():
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]

    for network in networks:
        ts = 17  # Example: creating 5 empty DataFrames
        runs = 100
        result = [[defaultdict for j in range(runs)] for i in range(ts)]
        for j in range(1, 101):
            print(j, network)
            with open("/home/ramesh/dblp/input/rait/" + network + "_tasks_summary_" + str(j) + ".txt", "r") as f:
                data = json.load(f)
                for key in data:
                    result[int(key) - 4][j - 1] = data[key]
        for ts in range(4, 21):
            df = pd.DataFrame(result[ts - 4], columns=[str(i) for i in range(ts + 1)])
            sns.boxplot(data=df)
            # plt.show()
            plt.xlabel("Number of popular skills")
            plt.ylabel("# Tasks")
            plt.savefig("/home/ramesh/dblp/output/eps/rait/" + network + "_" + str(ts) + ".eps", format="eps")
            plt.close()


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
    import pandas as pd
    # networks = ["colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
    #             "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "db", "dm", "th"]
    networks = ["vldb", "ai"]
    for network in networks:
        print(network)
        open("/home/ramesh/dblp/output/" + network + "_aco_teams.txt", "w").close()
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        raw_data = {nd1: {nd2: 0 for nd2 in graph.nodes} for nd1 in graph.nodes}
        distances = pd.DataFrame(raw_data, index=pd.Index([nd2 for nd2 in graph.nodes], name='RE'),
                                 columns=pd.Index([nd2 for nd2 in graph.nodes], name='CE'))
        for nd1 in list(graph.nodes):
            for nd2 in list(graph.nodes):
                distances.loc[nd1, nd2] = nx.dijkstra_path_length(graph, nd1, nd2, weight='weight')
        tasks = []
        with open("/home/ramesh/dblp/input/" + network + "_tasks.txt") as file:
        # with open("/home/ramesh/dblp/input/rait/" + network + "ar_tasks.txt") as file:
            for line in file:
                task = [x.strip() for x in line.strip("\n").split("\t") if x]
                tasks.append(task)
        popularity = dict()  # experts_for_skill i.e. skill:list of experts
        for node in graph.nodes:
            if "skills" in graph.nodes[node]:
                nskills = graph.nodes[node]["skills"].split(", ")
                nskills = {skl.strip() for skl in nskills if len(skl) > 0}
                for skill in nskills:
                    if skill in popularity:
                        popularity[skill.strip()].append(node)
                    else:
                        popularity[skill.strip()] = list()
                        popularity[skill.strip()].append(node)
        uskills = set()  # Universal set of skills
        expert_skills = {node: set() for node in graph.nodes}  # Dictionary:expert(key):list of skills(value) as strings
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(", "))
                expert_skills[node] = skls
                uskills.update(skls)
        blend = {skill: set() for skill in uskills}  # Blend value of skills, dictionary
        support = {skill: set() for skill in uskills}
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(", "))
                for skl in skls:
                    blend[skl.strip()].update(skls)
                    support[skl.strip()].add((node))
        for task in tasks:
            # with open("/home/ramesh/dblp/output/" + network + "_mpplrt_teams.txt", "a") as file:
            #     team, ptime = algorithms.pplrtdtfp(graph, task, popularity)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_TPLR11_teams.txt", "a") as file:
            #     team, rndm, ptime = algorithms.TPLRandom(graph, task, popularity_skill, 1, 1)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join(
            #         '%s,%s' % x for x in rndm) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_TPLR22_teams.txt", "a") as file:
            #     team, rndm, ptime = algorithms.TPLRandom(graph, task, popularity_skill, 2, 2)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join(
            #         '%s,%s' % x for x in rndm) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_TPLC11_teams.txt", "a") as file:
            #     team, rndm, ptime = algorithms.TPLClosest(graph, task, popularity_skill, 1, 1)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join(
            #         '%s,%s' % x for x in rndm) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_TPLC22_teams.txt", "a") as file:
            #     team, rndm, ptime = algorithms.TPLClosest(graph, task, popularity_skill, 2, 2)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + ",".join(
            #         '%s,%s' % x for x in rndm) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_minLD_teams.txt", "a") as file:
            #     team, ptime = algorithms.minLD(graph, task, popularity_skill)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_minSD_teams.txt", "a") as file:
            #     team, ptime = algorithms.minSD(graph, task, popularity_skill)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_rarestfirst_teams.txt", "a") as file:
            #     team, ptime = algorithms.rarestfirst(graph, task, popularity_skill)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            ## with open("/home/ramesh/dblp/output/rait/" + network + "_ar_genetic_teams.txt", "a") as file:
            ##     team, ptime = algorithms.genetic_algo(graph, task, popularity_skill)
            ##     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            ## with open("/home/ramesh/dblp/output/rait/" + network + "_ar_cultural_teams.txt", "a") as file:
            ##     team, ptime = algorithms.cultural(graph, task, popularity_skill)
            ##     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_popular_teams.txt", "a") as file:
            #     team, ptime = algorithms.pplrtdtfp(graph, task, popularity_skill)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            # with open("/home/ramesh/dblp/output/rait/" + network + "_ar_blend_teams.txt", "a") as file:
            #     team, ptime = algorithms.blenddtfp(graph, task, popularity_skill)
            #     file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            with open("/home/ramesh/dblp/output/" + network + "_aco_teams.txt", "a") as file:
                team, ptime = algorithms.aco(graph, task, popularity, distances)
                file.write(str(ptime) + "\t" + "\t".join('%s\t%s' % x for x in team) + "\n")
            print(len(task))


def make_results():
    networks = ["colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    # networks = ["vldb", "ai"]
    ts = [i for i in range(4, 21)]
    for network in networks:
        for algo in ["blend", "aco", "nsga2", "nsga3", "popular", "cmopso", "hybrid"]:
            graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt", "w").close()
            print(network, algo)
            try :
                with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_teams.txt") as file:
                    lc = 0
                    i = 0
                    for line in file:
                        lc += 1
                        result = [x for x in line.strip("\n").split("\t") if x]
                        if len(result) > 0:
                            team = [(result[1], "")]
                            for k in range(len(result)):
                                if k > 1 and k % 2 == 0:
                                    team.append(((result[k], result[k + 1].strip())))
                            task = [result[k].strip() for k in range(len(result)) if k > 2 and k % 2 != 0]
                            ld = algorithms.leader_distance(graph, team)
                            sd = algorithms.sum_distance(graph, team, task)
                            dd = algorithms.diameter_distance(graph, team)
                            tms = len(set([mmbr[0] for mmbr in team]))
                            gama = algorithms.gamma_diversity(graph, team)
                            shn = algorithms.shannon_diversity(graph, team)
                            igsn = algorithms.inverse_gini_simpson_diversisty(graph, team)
                            with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt",
                                      "a") as file1:
                                file1.write(
                                    str(ts[i]) + "\t" + result[0] + "\t" + str(tms) + "\t" + str(dd) + "\t" + str(ld) +
                                    "\t" + str(sd) + "\t" + str(gama) + "\t" + str(shn) + "\t" + str(igsn) + "\n")
                            if lc % 5 == 0:
                                i += 1
            except Exception as e:
                print("The error is: ", e)


def analysis():
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    # ts = [i for i in range(4, 21)]
    # networks = ["vldb", "ai"]
    j = 0
    for network in networks:
        for algo in ["blend", "aco", "nsga2", "nsga3", "popular", "cmopso", "hybrid"]:
            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_analysis.txt", "w").close()
            print(network, algo)
            try:
                with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt", "r") as file:
                    lc = 0
                    fsum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for line in file:
                        lc += 1
                        result = [x for x in line.strip("\n").split("\t") if x]
                        tsum = [fsum[i] + float(result[i]) for i in range(len(result))]
                        fsum = tsum
                        if lc % 5 == 0:
                            j += 1
                            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_analysis.txt", "a").write(
                                "\t".join([str(fsum[i] / 5) for i in range(len(result))]) + "\n")
                            fsum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                            tsum.clear()
            except Exception as e:
                print("The error is: ",e)


def network_details():
    # networks = ["nature", "physica", "science"]
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        uskills = set()
        # print(network)
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        avg_degree = (2 * nx.number_of_edges(graph)) / nx.number_of_nodes(graph)
        h1 = set()
        h2 = set()
        skill_popularity = dict()  # experts_for_skill i.e. skill:list of experts
        popularity_summary = dict()
        for node in list(graph.nodes):
            if len(graph.nodes[node]) > 0 and "skills" in graph.nodes[node]:
                skls = set(graph.nodes[node]["skills"].split(","))
                uskills.update(skls)
                uskills = {skl.strip() for skl in uskills if len(skl) > 0}
                for skill in graph.nodes[node]["skills"].split(", "):
                    if len(skill) > 0:
                        if skill in skill_popularity:
                            skill_popularity[skill].append(node)
                        else:
                            skill_popularity[skill] = list()
                            skill_popularity[skill].append(node)
            if graph.degree[node] > avg_degree:
                h1.add(node)
            if graph.degree[node] > (2 * avg_degree):
                h2.add(node)
        with open("/home/ramesh/dblp/output/" + network + "_skill_popularity.txt", "w") as file:
            for key in sorted(skill_popularity.keys()):
                if len(key) > 0:
                    file.write(str(key) + "\t" + ",".join(sorted(skill_popularity[key])) + "\n")
                    if len(skill_popularity[key]) in popularity_summary:
                        popularity_summary[len(skill_popularity[key])] += 1
                    else:
                        popularity_summary[len(skill_popularity[key])] = 1
                else:
                    pass
        num = 0
        den = 0
        with open("/home/ramesh/dblp/output/" + network + "_popularity_summary.txt", "w") as file:
            for key in sorted(popularity_summary.keys()):
                file.write(str(key) + "\t" + str(popularity_summary[key]) + "\n")
                num += (key * popularity_summary[key])
                den += popularity_summary[key]
        print(network.upper(), nx.number_of_nodes(graph), nx.number_of_edges(graph), len(uskills), nx.diameter(graph),
              round(avg_degree, 2), len(h1), len(h2), round(len(h1) / nx.number_of_nodes(graph), 2),
              round(len(h2) / nx.number_of_nodes(graph), 2), round((num / den), 2))


# ICDT 316 685 78 13 4.34 117 27 0.37 0.09 3.04
# COLT 624 1339 238 13 4.29 173 74 0.28 0.12 4.0
# PODS 712 1596 189 14 4.48 218 71 0.31 0.1 4.11
# PKDD 956 2206 164 25 4.62 333 84 0.35 0.09 4.42
# ECML 962 2104 183 23 4.37 315 98 0.33 0.1 4.2
# SDM 1033 2624 187 15 5.08 258 73 0.25 0.07 4.38
# STACS 1040 2047 233 28 3.94 449 106 0.43 0.1 3.21
# UAI 1141 2227 298 18 3.9 412 119 0.36 0.1 4.5
# EDBT 1659 4015 202 20 4.84 632 150 0.38 0.09 5.19
# STOC 1748 4792 481 13 5.48 507 235 0.29 0.13 5.93
# SODA 2541 6813 601 15 5.36 735 300 0.29 0.12 5.51
# FOCS 1773 4522 454 14 5.1 477 196 0.27 0.11 5.37
# ICML 2725 5977 355 25 4.39 787 269 0.29 0.1 6.88
# ICDM 2929 7687 392 24 5.25 825 247 0.28 0.08 7.08
# VLDB 3261 9912 494 13 6.08 879 321 0.27 0.1 8.02
# WWW 4203 11830 393 21 5.63 1387 362 0.33 0.09 8.05
# KDD 3316 10753 416 15 6.49 916 347 0.28 0.1 6.81
# SIGMOD 4856 20034 608 15 8.25 1419 468 0.29 0.1 8.42
# ICDE 4620 13638 497 15 5.9 1467 470 0.32 0.1 8.81
# AI 5708 13410 794 17 4.7 1770 568 0.31 0.1 9.83
# TH 5247 16997 1218 15 6.48 1400 638 0.27 0.12 11.49
# DB 11699 46127 1403 14 7.89 3305 1215 0.28 0.1 17.79
# DM 12993 40138 1199 20 6.18 3472 1221 0.27 0.09 14.54
# DBLP 32543 115938 2675 20 7.13 8149 3329 0.25 0.1 26.15


def powerlaw():
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    for network in networks:
        graph = nx.read_gml("/home/ramesh/dblp/input/" + network + ".gml")
        avg_degree = (2 * nx.number_of_edges(graph)) / nx.number_of_nodes(graph)
        high = dict()
        low = dict()
        for node in list(graph.nodes):
            if graph.degree[node] > avg_degree:
                if graph.degree[node] in high:
                    high[graph.degree[node]].update([node])
                else:
                    high[graph.degree[node]] = set()
                    high[graph.degree[node]].update([node])
            else:
                if graph.degree[node] in low:
                    low[graph.degree[node]].update([node])
                else:
                    low[graph.degree[node]] = set()
                    low[graph.degree[node]].update([node])
        open("/home/ramesh/dblp/output/" + network + "_hc.txt", "w").close()
        open("/home/ramesh/dblp/output/" + network + "_lc.txt", "w").close()
        open("/home/ramesh/dblp/output/" + network + "_nodes.txt", "w").close()
        with open("/home/ramesh/dblp/output/" + network + "_hc.txt", "a") as file1, open(
                "/home/ramesh/dblp/output/" + network + "_nodes.txt", "a") as file2:
            for key in high:
                file1.write(str(key) + "\t" + str(len(high[key])) + "\n")
                file2.write(str(key) + "\t" + str(len(high[key])) + "\n")
        with open("/home/ramesh/dblp/output/" + network + "_lc.txt", "a") as file1, open(
                "/home/ramesh/dblp/output/" + network + "_nodes.txt", "a") as file2:
            for key in low:
                file1.write(str(key) + "\t" + str(len(low[key])) + "\n")
                file2.write(str(key) + "\t" + str(len(low[key])) + "\n")

def view_toy():



if __name__ == '__main__':
    # generate_tasks()
    # experiment()
    make_results()
    analysis()
    # network_details()
    # powerlaw()
    # generate_rait_tasks()
    # rait_summary()
