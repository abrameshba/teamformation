import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class TeamSolution:
    team: List[Tuple[int,str]]  # Selected node IDs
    diameter_cost: float  # Communication diameter
    total_skill: float  # Total skill coverage score
    dominated: bool = False
    crowding_distance: float = 0.0


class CompetitiveMOPSO:
    def __init__(self, graph: nx.Graph, task: Set[str], popularity,
                 swarm_size: int = 100, archive_size: int = 50):
        self.graph = graph
        self.task = task
        self.popularity = popularity
        self.num_required_skills = len(task)
        self.nodes = list(graph.nodes())
        self.swarm_size = swarm_size
        self.archive_size = archive_size
        self.archive: List[TeamSolution] = []
        self.max_velocity_rate = 0.2
        self.mutation_rate = 0.3
        self.c1, self.c2 = 2.0, 2.0  # Cognitive/social coefficients [web:64]


    def initialize_swarm(self) -> List[TeamSolution]:
        """Initialize diverse swarm."""
        swarm = []
        for _ in range(self.swarm_size * 2):  # Generate extra for selection
            team = self.random_team()
            diam, skill = self.evaluate_team(team)
            swarm.append(TeamSolution(team, diam, skill))

        # Select best non-dominated solutions
        non_dominated = []
        for sol in swarm:
            if not any(self.dominates(other, sol) for other in swarm if other != sol):
                non_dominated.append(sol)

        return non_dominated[:self.swarm_size]

    def random_team(self) -> List[int]:
        """Generate random feasible team covering all 4 skills."""
        team = []
        for s in self.task:
            team.append((random.choice(list(self.popularity[s])), s))
        return team

    def gamma_diversity(self, team):
        gama = set()
        for nd in team:
            gama.update(set(self.graph.nodes[nd[0]]["skills"].split(", ")))
        return len(gama)

    def diameter_distance(self, team) -> float:
        team_nodes = set()
        leader = team[0][0]
        team_nodes.add(leader)
        uniq_members = set([mmbr[0] for mmbr in team])
        for expert in uniq_members:
            for node in nx.dijkstra_path(self.graph, leader, expert, weight="weight"):
                team_nodes.add(node)
        team_graph = nx.subgraph(self.graph, team_nodes).copy()
        return nx.diameter(team_graph)

    def evaluate_team(self, team: List[int]) -> Tuple[float, float]:
        """Evaluate team: (diameter_cost, total_skill_score)."""
        unique_nodes = list(set(team))

        if len(unique_nodes) == 0:
            return float('inf'), 0.0

        # Objective 1: Diameter (max shortest path weight between any pair)
        diameter = self.diameter_distance(team)

        # Objective 2: Total skill coverage score
        total_skill = self.gamma_diversity(team)

        return diameter, total_skill

    def dominates(self, sol1: TeamSolution, sol2: TeamSolution) -> bool:
        """sol1 dominates sol2 if better in ALL objectives."""
        return (sol1.diameter_cost < sol2.diameter_cost and
                sol1.total_skill > sol2.total_skill)

    def update_archive(self, new_solutions: List[TeamSolution]) -> None:
        """Competitive archive update with crowding distance."""
        candidates = self.archive + new_solutions

        # Find non-dominated solutions
        non_dominated = []
        for i, sol in enumerate(candidates):
            sol.dominated = any(self.dominates(other, sol)
                                for j, other in enumerate(candidates) if i != j)
            if not sol.dominated:
                non_dominated.append(sol)

        # Compute crowding distance
        if len(non_dominated) > 1:
            # Sort by objectives
            sorted_diam = sorted(non_dominated, key=lambda s: s.diameter_cost)
            sorted_skill = sorted(non_dominated, key=lambda s: -s.total_skill)

            # Assign crowding distances
            for sol in non_dominated:
                sol.crowding_distance = 0.0

            # Diameter crowding
            min_diam, max_diam = sorted_diam[0].diameter_cost, sorted_diam[-1].diameter_cost
            if max_diam > min_diam:
                for i, sol in enumerate(sorted_diam):
                    if i == 0 or i == len(sorted_diam) - 1:
                        sol.crowding_distance += float('inf')
                    else:
                        sol.crowding_distance += (sorted_diam[i + 1].diameter_cost -
                                                  sorted_diam[i - 1].diameter_cost) / (max_diam - min_diam)

            # Skill crowding
            min_skill, max_skill = sorted_skill[0].total_skill, sorted_skill[-1].total_skill
            if max_skill > min_skill:
                for i, sol in enumerate(sorted_skill):
                    if i == 0 or i == len(sorted_skill) - 1:
                        sol.crowding_distance += float('inf')
                    else:
                        sol.crowding_distance += (sorted_skill[i - 1].total_skill -
                                                  sorted_skill[i + 1].total_skill) / (max_skill - min_skill)

        # Competitive selection: keep diverse archive
        self.archive = sorted(non_dominated,
                              key=lambda s: (s.dominated, -s.crowding_distance))[:self.archive_size]

    def tournament_selection(self, candidates: List[TeamSolution], k: int = 3) -> TeamSolution:
        """Competitive tournament selection."""
        sample = random.sample(candidates, min(k, len(candidates)))
        # Prefer non-dominated with higher crowding distance
        return max(sample, key=lambda s: (0 if s.dominated else 1, s.crowding_distance))

    def generate_velocity(self, current_team: List[int], leader_team: List[int]) -> List[Tuple[int, int]]:
        """Generate swap-based velocity toward leader."""
        velocity = []
        current_set = set(current_team)
        leader_set = set(leader_team)
        # p_weight, g_weight = self.c1 / 2, self.c2 / 2

        # Add leader nodes not in current team
        for leader_node in leader_set - current_set:
            velocity.append((leader_node, 1.0))  # (node_id, strength)

        # Remove current nodes not in leader
        for current_node in current_set - leader_set:
            velocity.append((current_node, -1.0))  # Remove
        import math
        return velocity[:math.ceil(self.max_velocity_rate * len(self.nodes))]  # Limit velocity

    def apply_velocity(self, team: List[int], velocity: List[Tuple[int, int]]) -> List[int]:
        """Apply velocity to current team."""
        new_team = team.copy()

        # Process additions/removals
        for node_id, strength in velocity:
            if strength > 0:  # Add
                if node_id not in new_team:
                    new_team.append(node_id)
            else:  # Remove
                if node_id in new_team:
                    new_team.remove(node_id)

        # Ensure skill coverage
        return self.random_team() if len(set(new_team)) == 0 else new_team

    def optimize(self, max_iter: int = 25, verbose: bool = True) -> List[TeamSolution]:
        """Main CM-MOPSO optimization loop."""
        import time
        start_time = time.time_ns()
        swarm = self.initialize_swarm()
        self.update_archive(swarm)
        pbests = swarm.copy()

        for iteration in range(max_iter):
            new_swarm = []
            print(iteration)
            for particle in swarm:
                # Competitive leader selection
                p_leader = self.tournament_selection(pbests)
                g_leader = self.tournament_selection(self.archive)

                # Velocity update (cognitive + social)
                p_velocity = self.generate_velocity(particle.team, p_leader.team)
                g_velocity = self.generate_velocity(particle.team, g_leader.team)

                combined_velocity = p_velocity[:5] + g_velocity[:5]  # Balance

                # Position update
                new_team = self.apply_velocity(particle.team, combined_velocity)
                new_diam, new_skill = self.evaluate_team(new_team)
                new_particle = TeamSolution(new_team, new_diam, new_skill)

                new_swarm.append(new_particle)

                # Update pbest
                if self.dominates(new_particle, particle):
                    if particle in pbests:
                        idx = pbests.index(particle)
                        pbests[idx] = new_particle
                    else:
                        pbests[len(pbests) - 1] = new_particle

            swarm = new_swarm
            self.update_archive(swarm)

            if verbose and iteration % 5 == 0:
                best = min(self.archive, key=lambda s: s.diameter_cost)
                print(f"Iter {iteration}: Best diam={best.diameter_cost:.3f}, "
                      f"skill={best.total_skill:.3f}, Archive={len(self.archive)}")

        return sorted(self.archive, key=lambda s: (s.diameter_cost, -s.total_skill)), time.time_ns() - start_time


# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

if __name__ == "__main__":
    networks = ["colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    # networks = ["icdt"]
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
                for skill in graph.nodes[node]["skills"].split(", "):
                    if len(skill) > 0:
                        if skill in popularity:
                            popularity[skill.strip()].add(node)
                        else:
                            popularity[skill.strip()] = set()
                            popularity[skill.strip()].add(node)
        open("/home/ramesh/dblp/output/" + network + "_cmopso_teams_all.txt", "w").close()
        i = 0
        for task in tasks:
            i += 1
            cmopso = CompetitiveMOPSO(graph, task, popularity, swarm_size=10, archive_size=5)
            pareto_solutions, ptime = cmopso.optimize(max_iter=25, verbose=True)

            with open("/home/ramesh/dblp/output/" + network + "_cmopso_teams_all.txt", "a") as file:
                for sol in pareto_solutions:
                    print(sol.team)
                    # Ensure that each element in team is a pair
                    formatted_team = "\t".join(f'{pair[0]}\t{pair[1]}' for pair in sol.team if len(pair) == 2)
                    file.write(f"{ptime}\t{formatted_team}\n")
                    with open("/home/ramesh/dblp/output/" + network + "_cmopso_fitness.txt", "a") as file1:
                        file1.write(f"{sol.diameter_cost}\t{sol.total_skill}\n")
                if i % 5 == 0:
                    file.write("\n")
            # # Visualization
            # plt.figure(figsize=(12, 5))
            #
            # plt.subplot(1, 2, 1)
            # diams = [s.diameter_cost for s in pareto_solutions]
            # skills = [s.total_skill for s in pareto_solutions]
            # plt.scatter(diams, skills, c='red', s=80, alpha=0.7, edgecolors='black')
            # plt.xlabel('Diameter Cost (Minimize)')
            # plt.ylabel('Total Skill Score (Maximize)')
            # plt.title('CM-MOPSO Pareto Front')
            # plt.grid(True, alpha=0.3)
            #
            # plt.subplot(1, 2, 2)
            # pos = nx.spring_layout(graph, k=3, iterations=50)
            # nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)
            # nx.draw_networkx_edges(graph, pos, width=2, alpha=0.6, edge_color='gray')
            # nx.draw_networkx_labels(graph, pos, font_size=8)
            # plt.title('Expert Communication Graph')
            # plt.axis('off')
            #
            # plt.tight_layout()
            # plt.show()
            # with open("/home/ramesh/dblp/output/" + network + "_cmopso_teams_all.txt", "a") as file:
            #     for team in final_population:
            #         print(team)
            #         # Ensure that each element in team is a pair
            #         formatted_team = "\t".join(f'{pair[0]}\t{pair[1]}' for pair in team if len(pair) == 2)
            #         file.write(f"{ptime}\t{formatted_team}\n")
            #     if i % 5 == 0:
            #         file.write("\n")
            # with open("/home/ramesh/dblp/output/" + network + "_cmopso_fitness.txt", "a") as file1:
            #     for tmvl in final_fitness:
            #         file1.write(f"{tmvl[0]}\t{tmvl[1]}\n")
            #     # import matplotlib.pyplot as plt
            #     # # Output the final solutions
            #     # for ind, fitness in zip(final_population, final_fitness):
            #     #     print(f"Selected Nodes: {ind}, Fitness (Max property, Min distance): {fitness}")
            #     #     # Plot results
            #     #     plt.scatter([ind[1] for ind in final_fitness], [ind[0] for ind in final_fitness])
            #     # plt.xlabel('Communication cost')
            #     # plt.ylabel('Gamma diversity')
            #     # plt.title('NSGA-II Results')
            #     # plt.show()
