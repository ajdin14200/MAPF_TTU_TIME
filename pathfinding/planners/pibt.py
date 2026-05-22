# pathfinding/planners/PIBT.py

import math
import time
from collections import deque

from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
from pathfinding.planners.utils.time_uncertainty_solution import TimeUncertaintySolution


class PIBTPlanner:
    """
    Classical PIBT planner for unit-time MAPF.

    - Ignores edge uncertainty.
    - Each move has duration 1.
    - Supports a single goal or an ordered list of goals per agent.
    - Returns a TimeUncertaintySolution.
    """

    def __init__(self, tu_problem):
        self.problem = tu_problem
        self.agents = sorted(self.problem.start_positions.keys())
        self.adj = self._build_adjacency()
        self.dist_cache = {}

    def find_solution(self, time_limit=30, max_steps=None, soc=False):
        start_time = time.time()

        if max_steps is None:
            max_steps = 4 * len(self.adj) * max(1, len(self.agents))

        positions = {
            a: self.problem.start_positions[a]
            for a in self.agents
        }

        goals = {
            a: self._as_goal_list(self.problem.goal_positions[a])
            for a in self.agents
        }

        goal_index = {a: 0 for a in self.agents}

        for a in self.agents:
            while goal_index[a] < len(goals[a]) and positions[a] == goals[a][goal_index[a]]:
                goal_index[a] += 1

        paths = {
            a: [((0, 0), positions[a])]
            for a in self.agents
        }

        priority = {a: 0 for a in self.agents}

        for t in range(max_steps):
            if time.time() - start_time > time_limit:
                return TimeUncertaintySolution.void_solution(999)

            if self._all_done(goal_index, goals):
                duration = time.time() - start_time
                return self._build_solution(paths, soc, duration)

            for a in self.agents:
                if goal_index[a] < len(goals[a]):
                    priority[a] += 1
                else:
                    priority[a] = 0

            order = sorted(self.agents, key=lambda a: (-priority[a], a))

            next_pos = {}
            reserved = {}

            for a in order:
                if a not in next_pos:
                    self._pibt(
                        agent=a,
                        parent=None,
                        positions=positions,
                        next_pos=next_pos,
                        reserved=reserved,
                        goal_index=goal_index,
                        goals=goals
                    )

            for a in self.agents:
                if a not in next_pos:
                    next_pos[a] = positions[a]

            positions = next_pos

            for a in self.agents:
                paths[a].append(((t + 1, t + 1), positions[a]))

                old_idx = goal_index[a]
                while goal_index[a] < len(goals[a]) and positions[a] == goals[a][goal_index[a]]:
                    goal_index[a] += 1

                if goal_index[a] > old_idx:
                    priority[a] = 0

        return TimeUncertaintySolution.void_solution(-1)

    def _pibt(self, agent, parent, positions, next_pos, reserved, goal_index, goals):
        candidates = self._ordered_candidates(agent, positions, goal_index, goals)

        for v in candidates:
            if v in reserved:
                continue

            if self._creates_edge_swap(agent, v, positions, next_pos):
                continue

            occupant = self._agent_at(v, positions)

            next_pos[agent] = v
            reserved[v] = agent

            if occupant is not None and occupant != agent and occupant not in next_pos:
                success = self._pibt(
                    agent=occupant,
                    parent=agent,
                    positions=positions,
                    next_pos=next_pos,
                    reserved=reserved,
                    goal_index=goal_index,
                    goals=goals
                )

                if not success:
                    del next_pos[agent]
                    del reserved[v]
                    continue

            return True

        return False

    def _ordered_candidates(self, agent, positions, goal_index, goals):
        current = positions[agent]

        candidates = list(self.adj[current])
        candidates.append(current)  # wait action

        if goal_index[agent] >= len(goals[agent]):
            target = goals[agent][-1]
        else:
            target = goals[agent][goal_index[agent]]

        dist = self._distances_to(target)

        candidates.sort(
            key=lambda v: (
                dist.get(v, math.inf),
                0 if v != current else 1
            )
        )

        return candidates

    def _creates_edge_swap(self, agent, candidate, positions, next_pos):
        for other, other_next in next_pos.items():
            if positions[other] == candidate and other_next == positions[agent]:
                return True
        return False

    @staticmethod
    def _agent_at(vertex, positions):
        for a, pos in positions.items():
            if pos == vertex:
                return a
        return None

    @staticmethod
    def _as_goal_list(goal):
        if isinstance(goal, list):
            return goal
        return [goal]

    @staticmethod
    def _all_done(goal_index, goals):
        for a in goals:
            if goal_index[a] < len(goals[a]):
                return False
        return True

    def _build_adjacency(self):
        adj = {}

        for v, edges in self.problem.edges_and_weights.items():
            adj.setdefault(v, set())

            for edge in edges:
                neigh = edge[0]
                adj[v].add(neigh)
                adj.setdefault(neigh, set()).add(v)

        return adj

    def _distances_to(self, target):
        if target in self.dist_cache:
            return self.dist_cache[target]

        dist = {target: 0}
        queue = deque([target])

        while queue:
            v = queue.popleft()

            for neigh in self.adj.get(v, []):
                if neigh not in dist:
                    dist[neigh] = dist[v] + 1
                    queue.append(neigh)

        self.dist_cache[target] = dist
        return dist

    def _build_solution(self, paths, soc, duration):
        solution = TimeUncertaintySolution()
        solution.is_solved = True
        solution.time_to_solve = duration


        for agent, path in paths.items():
            cost = path[-1][0]
            solution.paths[agent] = TimeUncertaintyPlan(agent, path, cost)

        solution.compute_solution_cost(sum_of_costs=soc)
        solution.create_movement_tuples()

        return solution