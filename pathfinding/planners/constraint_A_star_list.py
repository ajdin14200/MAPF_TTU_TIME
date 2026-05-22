from pathfinding.planners.utils.custom_heap import OpenListHeap
from pathfinding.planners.utils.time_uncertainty_plan import TimeUncertaintyPlan
from pathfinding.planners.utils.time_error import OutOfTimeError

import math
import networkx
import time

# The positions of each parameter in the tuple receives from map.edges
VERTEX_ID = 0
STAY_STILL_COST = 1


class ConstraintAstar:

    def __init__(self, tu_problem):
        self.tu_problem = tu_problem
        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = -1

    """
    Computes the path for a particular agent in a given node. The node should contain the new constraint for this
    agents. Note that the agent is simply an int (the agent's id).

    returns a tuple containing (agent_id, cost, [ path list])

    *** Currently naive implementation (basic A* with constraints)****
    """

    def compute_agent_path(self, constraints, agent, start_pos, goals, conf_table,
                           mbc=True, time_limit=5, curr_time=(0, 0),
                           pos_cons=None, suboptimal=False):

        self.open_list = OpenListHeap()
        self.open_dict = {}
        self.closed_list = set()
        self.agent = agent

        # backward compatibility: allow a single goal too
        if not isinstance(goals, list):
            goals = [goals]

        start_time = time.time()

        # If the start position is already on one or more initial goals,
        # mark them as completed immediately.
        start_goal_index = 0
        while start_goal_index < len(goals) and start_pos == goals[start_goal_index]:
            start_goal_index += 1

        start_node = SingleAgentNode(
            start_pos,
            None,
            curr_time,
            self.tu_problem,
            goals,
            confs_created=0,
            goal_index=start_goal_index
        )
        self.__add_node_to_open(start_node, mbc)

        while len(self.open_list.internal_heap) > 0:
            if time.time() - start_time > time_limit:
                return TimeUncertaintyPlan.get_empty_plan(agent)

            best_node = self.open_list.pop()

            if best_node.goal_index == len(goals) and \
                    self.__can_stay(agent, best_node, constraints, suboptimal):
                return best_node.calc_path(agent)

            successors = best_node.expand(
                agent, constraints, conf_table, self.tu_problem, pos_cons, suboptimal
            )

            self.__remove_node_from_open(best_node)

            for neighbor in successors:
                if neighbor in self.closed_list:
                    continue

                g_val = neighbor[1]

                if neighbor not in self.open_dict:
                    neighbor_node = SingleAgentNode(
                        neighbor[0],      # current_position
                        best_node,        # prev_node
                        neighbor[1],      # g_val
                        self.tu_problem,
                        goals,
                        neighbor[2],      # confs_created
                        goal_index=neighbor[3]
                    )
                    self.__add_node_to_open(neighbor_node, mbc)
                else:
                    neighbor_node = self.open_dict[neighbor]
                    if mbc and g_val[0] >= neighbor_node.g_val[0]:
                        continue
                    elif not mbc and g_val[1] >= neighbor_node.g_val[1]:
                        continue

                    self.__update_node(
                        neighbor_node, best_node, g_val, goals, self.tu_problem,
                        neighbor[3]
                    )

        return TimeUncertaintyPlan.get_empty_plan(agent)

    def __remove_node_from_open(self, node):
        node_tuple = node.create_tuple()
        self.open_dict.pop(node_tuple, None)
        self.closed_list.add(node_tuple)

    def __add_node_to_open(self, node, min_best_case):
        if min_best_case:  # minimize the lower time bound
            self.open_list.push(node, node.g_val[0] + node.h_val, node.confs_created, -node.g_val[0])
        else:  # minimize the upper time bound
            self.open_list.push(node, node.g_val[1] + node.h_val, node.confs_created, -node.g_val[1])
        key_tuple = node.create_tuple()
        self.open_dict[key_tuple] = node

        return node

    @staticmethod
    def __update_node(neighbor_node, prev_node, g_val, goals, grid_map, goal_index):
        neighbor_node.prev_node = prev_node
        neighbor_node.g_val = g_val
        neighbor_node.goal_index = goal_index
        neighbor_node.h_val = neighbor_node.calc_sequence_heuristic(
            grid_map,
            neighbor_node.current_position,
            goals,
            goal_index
        )
        neighbor_node.f_val = (
            g_val[0] + neighbor_node.h_val,
            g_val[1] + neighbor_node.h_val
        )

    def __can_stay(self, agent, best_node, constraints, suboptimal):
        """
        A function that verifies that the agent has reached the goal at an appropriate time. Useful when an agent
        reaches the goal in, for example, 5-8 time units however in the solution it takes another agent 10-15 time
        units.
        Therefor we must verify what happens if this agent stands still all this time (there might be a conflict!)
        :param agent: The agent that's searching
        :param best_node: The current path
        :param constraints: A set of constraints
        :param suboptimal: If true, constraints apply to all agents and not the typical (agent, vertex, time) setting.
        this means that constraints are just (vertex, time) and apply to all agents. Used in prioritized planning since
        there is no need to iterate over all of the constraints.
        :return: True if the agent can stay in the current place, False otherwise.
        """
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
                    return False  # A constraint was found

        return True

    def dijkstra_solution(self, source_vertex, min_best_case=False):
        """
        A function that calculates, using Dijkstra's algorithm, the distance from each point in the map to the given
        goal. This will be used as a perfect heuristic. It receives as input the goal position and returns a dictionary
        mapping each coordinate to the distance from it to the goal.

        For now we will use the minimum time taken to pass an edge as the weight, in order to keep the heuristic
        admissible.
        """

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
        """ Returns true if the time intervals in 'time_1' and 'time_2' overlap.

        1: A<-- a<-->b -->B
        2: a<-- A -->b<---B>
        3: A<-- a<--- B --->b
        4: a<-- A<--->B --->b
        5: A<-->B
           a<-->b
        ===> A <= a <= B or a <= A <= b
        """
        return (time_1[0] <= time_2[0] <= time_1[1]) or (time_2[0] <= time_1[0] <= time_2[1])

    def __create_wait_node(self, best_node, con_time):
        curr_pos = best_node.current_position

        temp_best = SingleAgentNode(
            curr_pos,
            best_node.prev_node,
            best_node.g_val,
            self.tu_problem,
            best_node.goals,
            best_node.confs_created,
            goal_index=best_node.goal_index
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
            goal_index=best_node.goal_index
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
                goal_index=best_node.goal_index
            )
            prev_node = curr_node

        best_node.prev_node = prev_node
        best_node.g_val = (g_val[0] + 1, g_val[1] + 1)


class SingleAgentNode:
    """
    goal_index:
        index of the next goal not yet completed
    """

    def __init__(self, current_position, prev_node, g, grid_map, goals,
                 confs_created, goal_index=0):

        self.current_position = current_position
        self.prev_node = prev_node
        self.g_val = g
        self.confs_created = confs_created

        self.goals = goals
        self.goal_index = goal_index

        self.h_val = self.calc_sequence_heuristic(
            grid_map, current_position, goals, goal_index
        )
        self.f_val = (
            self.g_val[0] + self.h_val,
            self.g_val[1] + self.h_val
        )

    @staticmethod
    def calc_sequence_heuristic(grid_map, current_position, goals, goal_index):
        """
        Admissible heuristic for ordered goals:
        distance from current position to the next goal,
        plus distances between remaining consecutive goals.
        """
        if goal_index >= len(goals):
            return 0

        h = grid_map.calc_heuristic(current_position, goals[goal_index])
        for i in range(goal_index, len(goals) - 1):
            h += grid_map.calc_heuristic(goals[i], goals[i + 1])
        return h

    def create_tuple(self):
        """
        Must include progress in the goal sequence.
        """
        return (
            self.current_position,
            self.g_val,
            self.confs_created,
            self.goal_index
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
        # Case 1: normal one-tick wait
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
            while next_goal_index < len(self.goals) and self.current_position == self.goals[next_goal_index]:
                next_goal_index += 1

            stay_still = (
                self.current_position,
                still_time,
                confs_created,
                next_goal_index
            )
            neighbors.append(stay_still)

        # ------------------------------------------------------------
        # Case 2: normal moves
        # ------------------------------------------------------------
        for edge_tuple in search_map.edges_and_weights[self.current_position]:
            successor_time = (
                self.g_val[0] + edge_tuple[1][0],
                self.g_val[1] + edge_tuple[1][1]
            )
            vertex = edge_tuple[VERTEX_ID]

            if self.legal_move(agent, vertex, successor_time, constraints, pos_cons, suboptimal):
                next_goal_index = self.goal_index

                while next_goal_index < len(self.goals) and vertex == self.goals[next_goal_index]:
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
                    next_goal_index
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
                    if (pres[0] <= succ_time[0] <= pres[1]) or (succ_time[0] <= pres[0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[1]) - max(succ_time[0], pres[0]) + 1

            if edge in locations:
                for pres in locations[edge]:
                    if (pres[0][0] <= succ_time[0] <= pres[0][1]) or (succ_time[0] <= pres[0][0] <= succ_time[1]):
                        new_confs += min(succ_time[1], pres[0][1]) - max(succ_time[0], pres[0][0]) + 1

        return new_confs

    def legal_move(self, agent, vertex, succ_time, constraints, pos_cons, suboptimal):
        edge = min(self.current_position, vertex), max(self.current_position, vertex)
        edge_time = self.calc_edge_time(succ_time)

        if suboptimal:
            if edge in constraints:
                for tick in constraints[edge]:
                    if (tick[0] <= edge_time[0] <= tick[1]) or (edge_time[0] <= tick[0] <= edge_time[1]):
                        return False
            if vertex in constraints:
                for tick in constraints[vertex]:
                    if (tick[0] <= succ_time[0] <= tick[1]) or (succ_time[0] <= tick[0] <= succ_time[1]):
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