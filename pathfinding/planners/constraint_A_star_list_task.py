'''
from pathfinding.planners.utils.custom_heap import OpenListHeap
from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
from pathfinding.planners.utils.time_error import OutOfTimeError

import math
import networkx
import time

VERTEX_ID = 0
STAY_STILL_COST = 1


class ConstraintAstar:

    def __init__(self, tu_problem):
        self.tu_problem = tu_problem
        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = -1

        # Debug counters
        self.expanded_nodes = 0
        self.generated_nodes = 0
        self.pending_action_expansions = 0

    def compute_agent_path(self, constraints, agent, start_pos, goals, conf_table,
                           subgoal_action_time=None, mbc=True, time_limit=5,
                           curr_time=(0, 0), pos_cons=None, suboptimal=False):

        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = agent

        self.expanded_nodes = 0
        self.generated_nodes = 0
        self.pending_action_expansions = 0

        if subgoal_action_time is None:
            subgoal_action_time = {}

        if not isinstance(goals, list):
            goals = [goals]

        start_time = time.time()

        start_goal_index = 0
        start_pending_action = False

        if len(goals) > 0 and start_pos == goals[0]:
            if len(goals) > 1:
                start_pending_action = True
            else:
                start_goal_index = 1

        start_node = SingleAgentNode(
            start_pos,
            None,
            curr_time,
            self.tu_problem,
            goals,
            confs_created=0,
            goal_index=start_goal_index,
            pending_goal_action=start_pending_action,
            subgoal_action_time=subgoal_action_time
        )

        self.__add_node_to_open(start_node, mbc)

        while len(self.open_list.internal_heap) > 0:
            if time.time() - start_time > time_limit:
                return TimeUncertaintyPlan.get_empty_plan(agent)

            best_node = self.open_list.pop()

            self.expanded_nodes += 1
            if best_node.pending_goal_action:
                self.pending_action_expansions += 1

            if best_node.goal_index == len(goals) and \
                    self.__can_stay(agent, best_node, constraints, suboptimal):
                return best_node.calc_path(agent)

            successors = best_node.expand(
                agent, constraints, conf_table, self.tu_problem, pos_cons, suboptimal
            )

            self.__remove_node_from_open(best_node)

            for neighbor in successors:
                self.generated_nodes += 1

                if neighbor in self.closed_list:
                    continue

                g_val = neighbor[1]

                if neighbor not in self.open_dict:
                    neighbor_node = SingleAgentNode(
                        neighbor[0],
                        best_node,
                        neighbor[1],
                        self.tu_problem,
                        goals,
                        neighbor[2],
                        goal_index=neighbor[3],
                        pending_goal_action=neighbor[4],
                        subgoal_action_time=subgoal_action_time
                    )
                    self.__add_node_to_open(neighbor_node, mbc)
                else:
                    neighbor_node = self.open_dict[neighbor]

                    if mbc and g_val[0] >= neighbor_node.g_val[0]:
                        continue
                    elif not mbc and g_val[1] >= neighbor_node.g_val[1]:
                        continue

                    self.__update_node(
                        neighbor_node,
                        best_node,
                        g_val,
                        goals,
                        self.tu_problem,
                        neighbor[3],
                        neighbor[4],
                        subgoal_action_time
                    )

        return TimeUncertaintyPlan.get_empty_plan(agent)

    def __remove_node_from_open(self, node):
        node_tuple = node.create_tuple()
        self.open_dict.pop(node_tuple, None)
        self.closed_list.add(node_tuple)

    def __add_node_to_open(self, node, min_best_case):
        if min_best_case:
            self.open_list.push(node, node.g_val[0] + node.h_val, node.confs_created, -node.g_val[0])
        else:
            self.open_list.push(node, node.g_val[1] + node.h_val, node.confs_created, -node.g_val[1])

        key_tuple = node.create_tuple()
        self.open_dict[key_tuple] = node
        return node

    @staticmethod
    def __update_node(neighbor_node, prev_node, g_val, goals, grid_map,
                      goal_index, pending_goal_action, subgoal_action_time):
        neighbor_node.prev_node = prev_node
        neighbor_node.g_val = g_val
        neighbor_node.goal_index = goal_index
        neighbor_node.pending_goal_action = pending_goal_action
        neighbor_node.subgoal_action_time = subgoal_action_time

        neighbor_node.h_val = neighbor_node.calc_sequence_heuristic(
            grid_map,
            neighbor_node.current_position,
            goals,
            goal_index,
            pending_goal_action,
            subgoal_action_time
        )

        neighbor_node.f_val = (
            g_val[0] + neighbor_node.h_val,
            g_val[1] + neighbor_node.h_val
        )

    def __can_stay(self, agent, best_node, constraints, suboptimal):
        if suboptimal:
            if best_node.current_position in constraints:
                for tick in sorted(constraints[best_node.current_position], key=lambda k: k[1]):
                    if best_node.g_val[0] <= tick[1]:
                        self.__create_wait_node(best_node, tick[1])
                        self.open_dict = {}
                        self.open_list = OpenListHeap()
                        return False
            return True

        if best_node.current_position in constraints:
            for con in constraints[best_node.current_position]:
                if con[0] == agent and best_node.g_val[0] <= con[1][1]:
                    return False

        return True

    def dijkstra_solution(self, source_vertex, min_best_case=False):
        graph = networkx.Graph()

        for vertex, edges in self.tu_problem.edges_and_weights.items():
            for edge in edges:
                if min_best_case:
                    graph.add_edge(vertex, edge[0], weight=edge[1][0])
                else:
                    graph.add_edge(vertex, edge[0], weight=edge[1][1])

        try:
            return networkx.single_source_dijkstra_path_length(graph, source_vertex)
        except ValueError:
            print("neg value wut")

    @staticmethod
    def overlapping(time_1, time_2):
        return (time_1[0] <= time_2[0] <= time_1[1]) or \
               (time_2[0] <= time_1[0] <= time_2[1])

    def __create_wait_node(self, best_node, con_time):
        curr_pos = best_node.current_position

        temp_best = SingleAgentNode(
            curr_pos,
            best_node.prev_node,
            best_node.g_val,
            self.tu_problem,
            best_node.goals,
            best_node.confs_created,
            goal_index=best_node.goal_index,
            pending_goal_action=best_node.pending_goal_action,
            subgoal_action_time=best_node.subgoal_action_time
        )

        delta = best_node.g_val[1] - best_node.g_val[0]
        g_val = (best_node.g_val[0] + 1, best_node.g_val[0] + 1 + delta)

        prev_node = SingleAgentNode(
            curr_pos,
            temp_best,
            g_val,
            self.tu_problem,
            best_node.goals,
            best_node.confs_created,
            goal_index=best_node.goal_index,
            pending_goal_action=best_node.pending_goal_action,
            subgoal_action_time=best_node.subgoal_action_time
        )

        for g in range(prev_node.g_val[0] + 1, con_time - 1):
            g_val = (g, g + delta)
            curr_node = SingleAgentNode(
                curr_pos,
                prev_node,
                g_val,
                self.tu_problem,
                best_node.goals,
                best_node.confs_created,
                goal_index=best_node.goal_index,
                pending_goal_action=best_node.pending_goal_action,
                subgoal_action_time=best_node.subgoal_action_time
            )
            prev_node = curr_node

        best_node.prev_node = prev_node
        best_node.g_val = (g_val[0] + 1, g_val[1] + 1)


class SingleAgentNode:

    def __init__(self, current_position, prev_node, g, grid_map, goals,
                 confs_created, goal_index=0, pending_goal_action=False,
                 subgoal_action_time=None):

        self.current_position = current_position
        self.prev_node = prev_node
        self.g_val = g
        self.confs_created = confs_created

        self.goals = goals
        self.goal_index = goal_index
        self.pending_goal_action = pending_goal_action
        self.subgoal_action_time = subgoal_action_time or {}

        self.h_val = self.calc_sequence_heuristic(
            grid_map,
            current_position,
            goals,
            goal_index,
            pending_goal_action,
            self.subgoal_action_time
        )

        self.f_val = (
            self.g_val[0] + self.h_val,
            self.g_val[1] + self.h_val
        )

    @staticmethod
    def calc_sequence_heuristic(grid_map, current_position, goals, goal_index,
                                pending_goal_action=False, subgoal_action_time=None):
        """
        Admissible heuristic for ordered goals with task durations.

        It includes:
        - distance from current position to next goal
        - distances between remaining goals
        - minimum task duration for each remaining non-final subgoal
        """

        if goal_index >= len(goals):
            return 0

        if subgoal_action_time is None:
            subgoal_action_time = {}

        h = grid_map.calc_heuristic(current_position, goals[goal_index])

        for i in range(goal_index, len(goals) - 1):
            subgoal = goals[i]

            if subgoal in subgoal_action_time:
                h += subgoal_action_time[subgoal][0]

            h += grid_map.calc_heuristic(goals[i], goals[i + 1])

        return h

    def create_tuple(self):
        return (
            self.current_position,
            self.g_val,
            self.confs_created,
            self.goal_index,
            self.pending_goal_action
        )

    def calc_path(self, agent):
        path = []
        curr_node = self

        while curr_node:
            move = (curr_node.g_val, curr_node.current_position)
            path.insert(0, move)
            curr_node = curr_node.prev_node

        return TimeUncertaintyPlan(agent, path, self.g_val)

    def _count_conflicts_for_successor(self, agent, conflict_table, succ_time, edge):
        if len(conflict_table) == 0:
            return 0
        return self.confs_created + self.count_conflicts(agent, conflict_table, succ_time, edge)

    def expand(self, agent, constraints, conflict_table, search_map, pos_cons, suboptimal):
        neighbors = []

        # ------------------------------------------------------------
        # Case 1: pending subgoal action
        # ------------------------------------------------------------
        if self.pending_goal_action:
            goal_vertex = self.goals[self.goal_index]

            assert self.current_position == goal_vertex, \
                "Pending subgoal action but not on the subgoal vertex"

            # Do NOT mutate self.subgoal_action_time here.
            action_duration = self.subgoal_action_time.get(goal_vertex, (1, 1))

            action_time = (
                self.g_val[0] + action_duration[0],
                self.g_val[1] + action_duration[1]
            )

            if self.legal_move(agent, self.current_position, action_time, constraints, pos_cons, suboptimal):
                confs_created = self._count_conflicts_for_successor(
                    agent,
                    conflict_table,
                    action_time,
                    (self.current_position, self.current_position)
                )

                successor = (
                    self.current_position,
                    action_time,
                    confs_created,
                    self.goal_index + 1,
                    False
                )
                neighbors.append(successor)

            return neighbors

        # ------------------------------------------------------------
        # Case 2: normal one-tick wait
        # ------------------------------------------------------------
        still_time = (
            self.g_val[0] + STAY_STILL_COST,
            self.g_val[1] + STAY_STILL_COST
        )

        if self.legal_move(agent, self.current_position, still_time, constraints, pos_cons, suboptimal):
            confs_created = self._count_conflicts_for_successor(
                agent,
                conflict_table,
                (still_time[1], still_time[1]),
                (self.current_position, self.current_position)
            )

            next_goal_index = self.goal_index
            next_pending_action = False

            if next_goal_index < len(self.goals) and self.current_position == self.goals[next_goal_index]:
                if next_goal_index < len(self.goals) - 1:
                    next_pending_action = True
                else:
                    next_goal_index += 1

            stay_still = (
                self.current_position,
                still_time,
                confs_created,
                next_goal_index,
                next_pending_action
            )
            neighbors.append(stay_still)

        # ------------------------------------------------------------
        # Case 3: normal moves
        # ------------------------------------------------------------
        for edge_tuple in search_map.edges_and_weights[self.current_position]:
            successor_time = (
                self.g_val[0] + edge_tuple[1][0],
                self.g_val[1] + edge_tuple[1][1]
            )

            vertex = edge_tuple[VERTEX_ID]

            if self.legal_move(agent, vertex, successor_time, constraints, pos_cons, suboptimal):
                next_goal_index = self.goal_index
                next_pending_action = False

                if next_goal_index < len(self.goals) and vertex == self.goals[next_goal_index]:
                    if next_goal_index < len(self.goals) - 1:
                        next_pending_action = True
                    else:
                        next_goal_index += 1

                confs_created = self._count_conflicts_for_successor(
                    agent,
                    conflict_table,
                    successor_time,
                    (self.current_position, vertex)
                )

                successor = (
                    vertex,
                    successor_time,
                    confs_created,
                    next_goal_index,
                    next_pending_action
                )
                neighbors.append(successor)

        return neighbors

    @staticmethod
    def count_conflicts(agent, conflict_table, succ_time, edge):
        new_confs = 0

        for other, locations in conflict_table.items():
            if other == agent:
                continue

            if edge[1] in locations:
                for pres in locations[edge[1]]:
                    if (pres[0] <= succ_time[0] <= pres[1]) or \
                       (succ_time[0] <= pres[0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[1]) - max(succ_time[0], pres[0]) + 1

            if edge in locations:
                for pres in locations[edge]:
                    if (pres[0][0] <= succ_time[0] <= pres[0][1]) or \
                       (succ_time[0] <= pres[0][0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[0][1]) - max(succ_time[0], pres[0][0]) + 1

        return new_confs

    def legal_move(self, agent, vertex, succ_time, constraints, pos_cons, suboptimal):
        edge = min(self.current_position, vertex), max(self.current_position, vertex)
        edge_time = self.calc_edge_time(succ_time)

        if suboptimal:
            if edge in constraints:
                for tick in constraints[edge]:
                    if (tick[0] <= edge_time[0] <= tick[1]) or \
                       (edge_time[0] <= tick[0] <= edge_time[1]):
                        return False

            if vertex in constraints:
                for tick in constraints[vertex]:
                    if (tick[0] <= succ_time[0] <= tick[1]) or \
                       (succ_time[0] <= tick[0] <= succ_time[1]):
                        return False

            return True

        if pos_cons:
            for tick in range(succ_time[0], succ_time[1] + 1):
                if (agent, vertex, tick) not in pos_cons:
                    return False

        if edge in constraints:
            for con in constraints[edge]:
                if con[0] == agent and (
                    (con[1][0] <= edge_time[0] <= con[1][1]) or
                    (edge_time[0] <= con[1][1] <= edge_time[1])
                ):
                    return False

        if vertex in constraints:
            for con in constraints[vertex]:
                if con[0] == agent and (
                    (con[1][0] <= succ_time[0] <= con[1][1]) or
                    (succ_time[0] <= con[1][1] <= succ_time[1])
                ):
                    return False

        return True

    def calc_edge_time(self, succ_time):
        if (succ_time[0] - self.g_val[0], succ_time[1] - self.g_val[1]) == (1, 1):
            return self.g_val[0], succ_time[1]
        return self.g_val[0], succ_time[1]
'''


from pathfinding.planners.utils.custom_heap import OpenListHeap
from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
import networkx
import time

VERTEX_ID = 0
STAY_STILL_COST = 1


class ConstraintAstar:

    def __init__(self, tu_problem):
        self.tu_problem = tu_problem
        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = -1

        self.expanded_nodes = 0
        self.generated_nodes = 0

    def compute_agent_path(self, constraints, agent, start_pos, goals, conf_table,
                           subgoal_action_time=None, mbc=True, time_limit=5,
                           curr_time=(0, 0), pos_cons=None, suboptimal=False):

        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = agent

        self.expanded_nodes = 0
        self.generated_nodes = 0

        if subgoal_action_time is None:
            subgoal_action_time = {}

        if not isinstance(goals, list):
            goals = [goals]

        start_time = time.time()

        start_goal_index = 0
        start_g_val = curr_time
        start_presence_time = curr_time

        # If the agent starts on an initial subgoal, execute the task implicitly.
        if len(goals) > 0 and start_pos == goals[0]:
            if len(goals) > 1:
                task_lb, task_ub = subgoal_action_time.get(start_pos, (1, 1))
                start_presence_time = (curr_time[0], curr_time[1] + task_ub)
                start_g_val = (curr_time[0] + task_lb, curr_time[1] + task_ub)
                start_goal_index = 1
            else:
                start_goal_index = 1

        start_node = SingleAgentNode(
            current_position=start_pos,
            prev_node=None,
            g=start_g_val,
            presence_time=start_presence_time,
            grid_map=self.tu_problem,
            goals=goals,
            confs_created=0,
            goal_index=start_goal_index,
            subgoal_action_time=subgoal_action_time
        )

        self.__add_node_to_open(start_node, mbc)

        while len(self.open_list.internal_heap) > 0:
            if time.time() - start_time > time_limit:
                return TimeUncertaintyPlan.get_empty_plan(agent)

            best_node = self.open_list.pop()
            self.expanded_nodes += 1

            if best_node.goal_index == len(goals) and \
                    self.__can_stay(agent, best_node, constraints, suboptimal):
                return best_node.calc_path(agent)

            successors = best_node.expand(
                agent, constraints, conf_table, self.tu_problem, pos_cons, suboptimal
            )

            self.__remove_node_from_open(best_node)

            for neighbor in successors:
                self.generated_nodes += 1

                if neighbor in self.closed_list:
                    continue

                g_val = neighbor[1]

                if neighbor not in self.open_dict:
                    neighbor_node = SingleAgentNode(
                        current_position=neighbor[0],
                        prev_node=best_node,
                        g=neighbor[1],
                        presence_time=neighbor[2],
                        grid_map=self.tu_problem,
                        goals=goals,
                        confs_created=neighbor[3],
                        goal_index=neighbor[4],
                        subgoal_action_time=subgoal_action_time
                    )
                    self.__add_node_to_open(neighbor_node, mbc)

                else:
                    neighbor_node = self.open_dict[neighbor]

                    if mbc and g_val[0] >= neighbor_node.g_val[0]:
                        continue
                    elif not mbc and g_val[1] >= neighbor_node.g_val[1]:
                        continue

                    self.__update_node(
                        neighbor_node,
                        best_node,
                        neighbor[1],
                        neighbor[2],
                        goals,
                        self.tu_problem,
                        neighbor[4],
                        subgoal_action_time
                    )

        return TimeUncertaintyPlan.get_empty_plan(agent)

    def __remove_node_from_open(self, node):
        node_tuple = node.create_tuple()
        self.open_dict.pop(node_tuple, None)
        self.closed_list.add(node_tuple)

    def __add_node_to_open(self, node, min_best_case):
        if min_best_case:
            self.open_list.push(
                node,
                node.g_val[0] + node.h_val,
                node.confs_created,
                -node.g_val[0]
            )
        else:
            self.open_list.push(
                node,
                node.g_val[1] + node.h_val,
                node.confs_created,
                -node.g_val[1]
            )

        self.open_dict[node.create_tuple()] = node
        return node

    @staticmethod
    def __update_node(neighbor_node, prev_node, g_val, presence_time,
                      goals, grid_map, goal_index, subgoal_action_time):

        neighbor_node.prev_node = prev_node
        neighbor_node.g_val = g_val
        neighbor_node.presence_time = presence_time
        neighbor_node.goal_index = goal_index
        neighbor_node.subgoal_action_time = subgoal_action_time

        neighbor_node.h_val = neighbor_node.calc_sequence_heuristic(
            grid_map,
            neighbor_node.current_position,
            goals,
            goal_index,
            subgoal_action_time
        )

        neighbor_node.f_val = (
            g_val[0] + neighbor_node.h_val,
            g_val[1] + neighbor_node.h_val
        )

    def __can_stay(self, agent, best_node, constraints, suboptimal):
        if suboptimal:
            if best_node.current_position in constraints:
                for tick in sorted(constraints[best_node.current_position], key=lambda k: k[1]):
                    if best_node.g_val[0] <= tick[1]:
                        return False
            return True

        if best_node.current_position in constraints:
            for con in constraints[best_node.current_position]:
                if con[0] == agent and best_node.g_val[0] <= con[1][1]:
                    return False

        return True

    def dijkstra_solution(self, source_vertex, min_best_case=False):
        graph = networkx.Graph()

        for vertex, edges in self.tu_problem.edges_and_weights.items():
            for edge in edges:
                if min_best_case:
                    graph.add_edge(vertex, edge[0], weight=edge[1][0])
                else:
                    graph.add_edge(vertex, edge[0], weight=edge[1][1])

        return networkx.single_source_dijkstra_path_length(graph, source_vertex)


class SingleAgentNode:

    def __init__(self, current_position, prev_node, g, presence_time,
                 grid_map, goals, confs_created, goal_index=0,
                 subgoal_action_time=None):

        self.current_position = current_position
        self.prev_node = prev_node

        # ready-to-leave interval
        self.g_val = g

        # interval during which this vertex may be occupied
        self.presence_time = presence_time

        self.confs_created = confs_created
        self.goals = goals
        self.goal_index = goal_index
        self.subgoal_action_time = subgoal_action_time or {}

        self.h_val = self.calc_sequence_heuristic(
            grid_map,
            current_position,
            goals,
            goal_index,
            self.subgoal_action_time
        )

        self.f_val = (
            self.g_val[0] + self.h_val,
            self.g_val[1] + self.h_val
        )

    @staticmethod
    def calc_sequence_heuristic(grid_map, current_position, goals,
                                goal_index, subgoal_action_time=None):
        if goal_index >= len(goals):
            return 0

        if subgoal_action_time is None:
            subgoal_action_time = {}

        h = grid_map.calc_heuristic(current_position, goals[goal_index])

        for i in range(goal_index, len(goals) - 1):
            subgoal = goals[i]

            if subgoal in subgoal_action_time:
                h += subgoal_action_time[subgoal][0]

            h += grid_map.calc_heuristic(goals[i], goals[i + 1])

        return h

    def create_tuple(self):
        return (
            self.current_position,
            self.g_val,
            self.presence_time,
            self.confs_created,
            self.goal_index
        )

    def calc_path(self, agent):
        path = []
        curr_node = self

        while curr_node:
            move = (curr_node.presence_time, curr_node.current_position)
            path.insert(0, move)
            curr_node = curr_node.prev_node

        return TimeUncertaintyPlan(agent, path, self.g_val)

    def _count_conflicts_for_successor(self, agent, conflict_table, vertex_time, edge):
        if len(conflict_table) == 0:
            return 0

        return self.confs_created + self.count_conflicts(
            agent,
            conflict_table,
            vertex_time,
            edge
        )

    def expand(self, agent, constraints, conflict_table, search_map, pos_cons, suboptimal):
        neighbors = []

        # ------------------------------------------------------------
        # Case 1: wait action
        # ------------------------------------------------------------
        still_g = (
            self.g_val[0] + STAY_STILL_COST,
            self.g_val[1] + STAY_STILL_COST
        )

        still_presence = still_g
        edge_time = (self.g_val[0], still_g[1])

        if self.legal_move(
            agent,
            self.current_position,
            still_presence,
            constraints,
            pos_cons,
            suboptimal,
            edge_time=edge_time
        ):
            confs_created = self._count_conflicts_for_successor(
                agent,
                conflict_table,
                still_presence,
                (self.current_position, self.current_position)
            )

            neighbors.append((
                self.current_position,
                still_g,
                still_presence,
                confs_created,
                self.goal_index
            ))

        # ------------------------------------------------------------
        # Case 2: movement action, possibly followed by implicit task
        # ------------------------------------------------------------
        for edge_tuple in search_map.edges_and_weights[self.current_position]:
            vertex = edge_tuple[VERTEX_ID]
            edge_lb, edge_ub = edge_tuple[1]

            arrival_time = (
                self.g_val[0] + edge_lb,
                self.g_val[1] + edge_ub
            )

            edge_time = (
                self.g_val[0],
                arrival_time[1]
            )

            next_goal_index = self.goal_index
            successor_g = arrival_time
            successor_presence = arrival_time

            # If this vertex is the next required non-final goal,
            # execute the task implicitly.
            if next_goal_index < len(self.goals) and vertex == self.goals[next_goal_index]:

                if next_goal_index < len(self.goals) - 1:
                    task_lb, task_ub = self.subgoal_action_time.get(vertex, (1, 1))

                    successor_presence = (
                        arrival_time[0],
                        arrival_time[1] + task_ub
                    )

                    successor_g = (
                        arrival_time[0] + task_lb,
                        arrival_time[1] + task_ub
                    )

                    next_goal_index += 1

                else:
                    next_goal_index += 1

            if self.legal_move(
                agent,
                vertex,
                successor_presence,
                constraints,
                pos_cons,
                suboptimal,
                edge_time=edge_time
            ):
                confs_created = self._count_conflicts_for_successor(
                    agent,
                    conflict_table,
                    successor_presence,
                    (self.current_position, vertex)
                )

                neighbors.append((
                    vertex,
                    successor_g,
                    successor_presence,
                    confs_created,
                    next_goal_index
                ))

        return neighbors

    @staticmethod
    def count_conflicts(agent, conflict_table, succ_time, edge):
        new_confs = 0

        for other, locations in conflict_table.items():
            if other == agent:
                continue

            if edge[1] in locations:
                for pres in locations[edge[1]]:
                    if (pres[0] <= succ_time[0] <= pres[1]) or \
                       (succ_time[0] <= pres[0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[1]) - max(succ_time[0], pres[0]) + 1

            if edge in locations:
                for pres in locations[edge]:
                    if (pres[0][0] <= succ_time[0] <= pres[0][1]) or \
                       (succ_time[0] <= pres[0][0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[0][1]) - max(succ_time[0], pres[0][0]) + 1

        return new_confs

    def legal_move(self, agent, vertex, vertex_time, constraints,
                   pos_cons, suboptimal, edge_time=None):

        edge = min(self.current_position, vertex), max(self.current_position, vertex)

        if edge_time is None:
            edge_time = (self.g_val[0], vertex_time[1])

        if suboptimal:
            if edge in constraints:
                for tick in constraints[edge]:
                    if self.overlap(edge_time, tick):
                        return False

            if vertex in constraints:
                for tick in constraints[vertex]:
                    if self.overlap(vertex_time, tick):
                        return False

            return True

        if pos_cons:
            for tick in range(vertex_time[0], vertex_time[1] + 1):
                if (agent, vertex, tick) not in pos_cons:
                    return False

        if edge in constraints:
            for con in constraints[edge]:
                if con[0] == agent and self.overlap(edge_time, con[1]):
                    return False

        if vertex in constraints:
            for con in constraints[vertex]:
                if con[0] == agent and self.overlap(vertex_time, con[1]):
                    return False

        return True

    @staticmethod
    def overlap(t1, t2):
        return (t1[0] <= t2[0] <= t1[1]) or \
               (t2[0] <= t1[0] <= t2[1])