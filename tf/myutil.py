import networkx as nx


def suputil():
    team = list()
    tot_skills = 0
    graph = nx.read_gml("/home/ramesh/dblp/input/vldb.gml")
    all_skills = set()
    with open("/home/ramesh/dblp/output/vldb_aco_teams.txt") as file:
        i=0
        for line in file:
            i+=1
            iwords = [word.strip() for word in line.strip("\n").split("\t") if len(word) > 1]
            if i>0 and i<6:
                # print(iwords)
                for j in range(len(iwords)):
                    if j>0 and j%2==0:
                        team.append(iwords[j])
                # print(team)
                for node in team:
                    if "skills" in graph.nodes[node]:
                        all_skills.update(graph.nodes[node]["skills"].split(","))
                        # print(len(all_skills), end="\t")
                tot_skills += len(all_skills)
                team.clear()
                all_skills.clear()
        print(tot_skills/5)



if __name__ == '__main__':
    suputil()
