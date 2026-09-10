"""Constraint A* for CBS_TTU with explicit uncertain task actions.

This version combines the useful parts of the two historical implementations in
``constraint_A_star_list_task.py``:

* the task is an explicit low-level action, represented by
  ``pending_goal_action``;
* ``g_val`` (ready-to-act / ready-to-leave interval) is kept distinct from
  ``presence_time`` (the interval during which the current vertex may be
  occupied).

The distinction is important at an intermediate goal.  If an agent arrives in
``[a-, a+]`` and executes a task of duration ``[p-, p+]``, then the state after
that task has

    presence_time = [a-, a+ + p+]
    g_val         = [a- + p-, a+ + p+]

so vertex conflicts see the whole task occupancy while subsequent traversal
starts no earlier than task completion.

The generated ``TimeUncertaintyPlan`` keeps the legacy ``path`` format
``[(presence_interval, vertex), ...]`` for compatibility and additionally gets
an aligned ``ready_times`` attribute containing the ``g_val`` of each state.
``cbs_ttu_fix.py`` uses this metadata when creating edge-movement intervals.
"""

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

        # Debug counters retained from the previous task-aware implementation.
        self.expanded_nodes = 0
        self.generated_nodes = 0
        self.pending_action_expansions = 0

    def compute_agent_path(
        self,
        constraints,
        agent,
        start_pos,
        goals,
        conf_table,
        subgoal_action_time=None,
        mbc=True,
        time_limit=5,
        curr_time=(0, 0),
        pos_cons=None,
        suboptimal=False,
    ):
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

        # If the start itself is the first intermediate goal, do not execute
        # the task implicitly.  The initial node is task-pending, so the task
        # goes through the normal legality/constraint checks below.
        if goals and start_pos == goals[0]:
            if len(goals) > 1:
                self._task_duration(goals[0], subgoal_action_time)
                start_pending_action = True
            else:
                start_goal_index = 1

        start_node = SingleAgentNode(
            current_position=start_pos,
            prev_node=None,
            g=curr_time,
            presence_time=curr_time,
            grid_map=self.tu_problem,
            goals=goals,
            confs_created=0,
            goal_index=start_goal_index,
            pending_goal_action=start_pending_action,
            subgoal_action_time=subgoal_action_time,
            incoming_action="start",
        )

        self.__add_node_to_open(start_node, mbc)

        while len(self.open_list.internal_heap) > 0:
            if time.time() - start_time > time_limit:
                return TimeUncertaintyPlan.get_empty_plan(agent)

            best_node = self.open_list.pop()

            # A better CAT tie-break path to the same temporal state may have
            # replaced an older heap entry.  Ignore such stale entries.
            if self.open_dict.get(best_node.state_key()) is not best_node:
                continue

            self.expanded_nodes += 1

            if best_node.pending_goal_action:
                self.pending_action_expansions += 1

            if (
                best_node.goal_index == len(goals)
                and not best_node.pending_goal_action
                and self.__can_stay(agent, best_node, constraints, suboptimal)
            ):
                return best_node.calc_path(agent)

            successors = best_node.expand(
                agent,
                constraints,
                conf_table,
                self.tu_problem,
                pos_cons,
                suboptimal,
            )

            self.__remove_node_from_open(best_node)

            for successor in successors:
                self.generated_nodes += 1
                state_key = successor.state_key()

                if state_key in self.closed_list:
                    continue

                current_open = self.open_dict.get(state_key)
                if current_open is None:
                    successor.prev_node = best_node
                    self.__add_node_to_open(successor, mbc)
                    continue

                # The temporal state is identical.  Keep the version that has
                # created fewer CAT conflicts; push it as a fresh heap entry
                # and let the stale-entry check above discard the old one.
                if successor.confs_created >= current_open.confs_created:
                    continue

                successor.prev_node = best_node
                self.__add_node_to_open(successor, mbc)

        return TimeUncertaintyPlan.get_empty_plan(agent)

    @staticmethod
    def _task_duration(vertex, subgoal_action_time):
        """Return a task interval and fail loudly for malformed TTU instances."""
        if vertex not in subgoal_action_time:
            raise ValueError(
                f"Missing task duration for intermediate goal {vertex!r}. "
                "Every non-final ordered goal must have a (lb, ub) task interval."
            )

        lb, ub = subgoal_action_time[vertex]
        if lb < 0 or ub < lb:
            raise ValueError(
                f"Invalid task duration for {vertex!r}: {(lb, ub)!r}"
            )
        return lb, ub

    def __remove_node_from_open(self, node):
        key = node.state_key()
        self.open_dict.pop(key, None)
        self.closed_list.add(key)

    def __add_node_to_open(self, node, min_best_case):
        if min_best_case:
            self.open_list.push(
                node,
                node.g_val[0] + node.h_val,
                node.confs_created,
                -node.g_val[0],
            )
        else:
            self.open_list.push(
                node,
                node.g_val[1] + node.h_val,
                node.confs_created,
                -node.g_val[1],
            )

        self.open_dict[node.state_key()] = node
        return node

    def __can_stay(self, agent, best_node, constraints, suboptimal):
        """Check that the agent may remain forever at its final vertex.

        This keeps the original CBS_TU semantics.  If a future constraint exists
        at the goal, A* continues expanding wait actions instead of accepting the
        node as a terminal solution.
        """
        if suboptimal:
            if best_node.current_position in constraints:
                for tick in sorted(
                    constraints[best_node.current_position], key=lambda k: k[1]
                ):
                    # Suboptimal constraints do not carry an agent id.
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
                weight = edge[1][0] if min_best_case else edge[1][1]
                graph.add_edge(vertex, edge[0], weight=weight)

        return networkx.single_source_dijkstra_path_length(graph, source_vertex)

    @staticmethod
    def overlapping(time_1, time_2):
        return (time_1[0] <= time_2[0] <= time_1[1]) or (
            time_2[0] <= time_1[0] <= time_2[1]
        )


class SingleAgentNode:
    def __init__(
        self,
        current_position,
        prev_node,
        g,
        presence_time,
        grid_map,
        goals,
        confs_created,
        goal_index=0,
        pending_goal_action=False,
        subgoal_action_time=None,
        incoming_action=None,
    ):
        self.current_position = current_position
        self.prev_node = prev_node

        # Earliest/latest time at which the agent is ready to execute its next
        # action (or leave this vertex).
        self.g_val = g

        # Potential occupancy of the current vertex represented by this state.
        # In a task-completion state this covers the whole task execution.
        self.presence_time = presence_time

        self.confs_created = confs_created
        self.goals = goals
        self.goal_index = goal_index
        self.pending_goal_action = pending_goal_action
        self.subgoal_action_time = subgoal_action_time or {}
        self.incoming_action = incoming_action

        self.h_val = self.calc_sequence_heuristic(
            grid_map,
            current_position,
            goals,
            goal_index,
            pending_goal_action,
            self.subgoal_action_time,
        )

        self.f_val = (
            self.g_val[0] + self.h_val,
            self.g_val[1] + self.h_val,
        )

    @staticmethod
    def calc_sequence_heuristic(
        grid_map,
        current_position,
        goals,
        goal_index,
        pending_goal_action=False,
        subgoal_action_time=None,
    ):
        """Admissible ordered-goal heuristic including task lower bounds."""
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

    def state_key(self):
        """Search-state identity requested by the explicit-task formulation.

        ``presence_time`` is included because two nodes may be equally ready to
        leave while representing different conservative occupancy intervals.
        CAT conflict counts are deliberately not part of the state identity;
        they are only a tie-breaker.
        """
        return (
            self.current_position,
            self.g_val,
            self.presence_time,
            self.goal_index,
            self.pending_goal_action,
        )

    # Backward-compatible name used by older surrounding code/debug tools.
    def create_tuple(self):
        return self.state_key()

    def calc_path(self, agent):
        path = []
        ready_times = []
        state_path = []

        curr_node = self
        while curr_node:
            path.insert(0, (curr_node.presence_time, curr_node.current_position))
            ready_times.insert(0, curr_node.g_val)
            state_path.insert(
                0,
                {
                    "position": curr_node.current_position,
                    "presence_time": curr_node.presence_time,
                    "g_val": curr_node.g_val,
                    "goal_index": curr_node.goal_index,
                    "pending_task": curr_node.pending_goal_action,
                    "incoming_action": curr_node.incoming_action,
                },
            )
            curr_node = curr_node.prev_node

        plan = TimeUncertaintyPlan(agent, path, self.g_val)
        # Extra metadata is intentionally additive so existing code that only
        # consumes ``plan.path`` keeps working.
        plan.ready_times = ready_times
        plan.state_path = state_path
        return plan

    def _count_conflicts_for_successor(
        self, agent, conflict_table, vertex_time, edge, edge_time=None
    ):
        if len(conflict_table) == 0:
            return self.confs_created

        return self.confs_created + self.count_conflicts(
            agent,
            conflict_table,
            vertex_time,
            edge,
            edge_time=edge_time,
        )

    def _wait_successor(
        self, agent, constraints, conflict_table, search_map, pos_cons, suboptimal
    ):
        """One deterministic wait, preserving task-pending status."""
        still_g = (
            self.g_val[0] + STAY_STILL_COST,
            self.g_val[1] + STAY_STILL_COST,
        )

        # The current state already covers its previous presence.  Including
        # the whole wait interval here is conservative and makes the action's
        # legality explicit.
        wait_presence = still_g #(self.g_val[0], still_g[1])
        edge_time = (self.g_val[0], still_g[1])

        if not self.legal_move(
            agent,
            self.current_position,
            wait_presence,
            constraints,
            pos_cons,
            suboptimal,
            edge_time=edge_time
        ):
            return None

        confs_created = self._count_conflicts_for_successor(
            agent,
            conflict_table,
            wait_presence,
            (self.current_position, self.current_position),
            edge_time=None,
        )

        return SingleAgentNode(
            current_position=self.current_position,
            prev_node=None,
            g=still_g,
            presence_time=wait_presence,
            grid_map=search_map,
            goals=self.goals,
            confs_created=confs_created,
            goal_index=self.goal_index,
            pending_goal_action=self.pending_goal_action,
            subgoal_action_time=self.subgoal_action_time,
            incoming_action="wait",
        )

    def expand(
        self, agent, constraints, conflict_table, search_map, pos_cons, suboptimal
    ):
        neighbors = []

        # Waiting is legal both before and after task execution.  This is the
        # key difference from the historical first implementation, where a
        # pending task had to start immediately.
        wait_node = self._wait_successor(
            agent, constraints, conflict_table, search_map, pos_cons, suboptimal
        )
        if wait_node is not None:
            neighbors.append(wait_node)

        # ------------------------------------------------------------
        # Explicit task action
        # ------------------------------------------------------------
        if self.pending_goal_action:
            goal_vertex = self.goals[self.goal_index]
            if self.current_position != goal_vertex:
                raise AssertionError(
                    "Pending subgoal action but not on the corresponding goal vertex"
                )

            task_lb, task_ub = ConstraintAstar._task_duration(
                goal_vertex, self.subgoal_action_time
            )
            task_done_g = (
                self.g_val[0] + task_lb,
                self.g_val[1] + task_ub,
            )

            # From task start until the latest possible completion, the robot
            # occupies the subgoal vertex.
            task_presence = task_done_g #(self.g_val[0], task_done_g[1])
            edge_time = (self.g_val[0], task_done_g[1])

            if self.legal_move(
                agent,
                self.current_position,
                task_presence,
                constraints,
                pos_cons,
                suboptimal,
                edge_time=edge_time
            ):
                confs_created = self._count_conflicts_for_successor(
                    agent,
                    conflict_table,
                    task_presence,
                    (self.current_position, self.current_position),
                    edge_time=None,
                )

                neighbors.append(
                    SingleAgentNode(
                        current_position=self.current_position,
                        prev_node=None,
                        g=task_done_g,
                        presence_time=task_presence,
                        grid_map=search_map,
                        goals=self.goals,
                        confs_created=confs_created,
                        goal_index=self.goal_index + 1,
                        pending_goal_action=False,
                        subgoal_action_time=self.subgoal_action_time,
                        incoming_action="task",
                    )
                )

            # While a task is pending the only choices are wait or execute it;
            # the agent cannot leave the subgoal before task completion.
            return neighbors

        # ------------------------------------------------------------
        # Movement actions
        # ------------------------------------------------------------
        for edge_tuple in search_map.edges_and_weights[self.current_position]:
            vertex = edge_tuple[VERTEX_ID]
            edge_lb, edge_ub = edge_tuple[1]

            arrival_time = (
                self.g_val[0] + edge_lb,
                self.g_val[1] + edge_ub,
            )
            edge_time = (self.g_val[0], arrival_time[1])

            next_goal_index = self.goal_index
            next_pending_action = False

            if next_goal_index < len(self.goals) and vertex == self.goals[next_goal_index]:
                if next_goal_index < len(self.goals) - 1:
                    ConstraintAstar._task_duration(vertex, self.subgoal_action_time)
                    next_pending_action = True
                else:
                    # Final goal has no task in MAPF-TTU.
                    next_goal_index += 1

            if not self.legal_move(
                agent,
                vertex,
                arrival_time,
                constraints,
                pos_cons,
                suboptimal,
                edge_time=edge_time,
                check_edge=True,
            ):
                continue

            confs_created = self._count_conflicts_for_successor(
                agent,
                conflict_table,
                arrival_time,
                (self.current_position, vertex),
                edge_time=edge_time,
            )

            neighbors.append(
                SingleAgentNode(
                    current_position=vertex,
                    prev_node=None,
                    g=arrival_time,
                    presence_time=arrival_time,
                    grid_map=search_map,
                    goals=self.goals,
                    confs_created=confs_created,
                    goal_index=next_goal_index,
                    pending_goal_action=next_pending_action,
                    subgoal_action_time=self.subgoal_action_time,
                    incoming_action="move",
                )
            )

        return neighbors

    @staticmethod
    def count_conflicts(
        agent, conflict_table, vertex_time, edge, edge_time=None
    ):
        """Count CAT overlaps for tie-breaking.

        Vertex occupancy uses ``vertex_time``.  Real movement edges use their
        own ``edge_time``; self-transitions (wait/task) intentionally have no
        edge occupancy because their collision semantics are purely vertex
        occupancy.
        """
        new_confs = 0

        for other, locations in conflict_table.items():
            if other == agent:
                continue

            if edge[1] in locations:
                for pres in locations[edge[1]]:
                    # CAT entries for vertices are plain intervals; edge CAT
                    # entries are ``(interval, direction)`` and therefore not
                    # considered in this branch.
                    if (
                        isinstance(pres, tuple)
                        and len(pres) == 2
                        and not isinstance(pres[0], tuple)
                    ):
                        if ConstraintAstar.overlapping(vertex_time, pres):
                            new_confs += (
                                min(vertex_time[1], pres[1])
                                - max(vertex_time[0], pres[0])
                                + 1
                            )

            if edge_time is not None:
                canonical_edge = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                if canonical_edge in locations:
                    for pres in locations[canonical_edge]:
                        if not (
                            isinstance(pres, tuple)
                            and len(pres) == 2
                            and isinstance(pres[0], tuple)
                        ):
                            continue
                        other_interval = pres[0]
                        if ConstraintAstar.overlapping(edge_time, other_interval):
                            new_confs += (
                                min(edge_time[1], other_interval[1])
                                - max(edge_time[0], other_interval[0])
                                + 1
                            )

        return new_confs

    def legal_move(
        self,
        agent,
        vertex,
        vertex_time,
        constraints,
        pos_cons,
        suboptimal,
        edge_time=None,
        check_edge=True,
    ):
        canonical_edge = (
            min(self.current_position, vertex),
            max(self.current_position, vertex),
        )

        if suboptimal:
            if check_edge and edge_time is not None and canonical_edge in constraints:
                for tick in constraints[canonical_edge]:
                    if ConstraintAstar.overlapping(edge_time, tick):
                        return False

            if vertex in constraints:
                for tick in constraints[vertex]:
                    if ConstraintAstar.overlapping(vertex_time, tick):
                        return False
            return True

        if pos_cons:
            # The original implementation assumes integer time.  Preserve that
            # contract here rather than silently rounding continuous values.
            for tick in range(vertex_time[0], vertex_time[1] + 1):
                if (agent, vertex, tick) not in pos_cons:
                    return False

        if check_edge and edge_time is not None and canonical_edge in constraints:
            for con in constraints[canonical_edge]:
                if con[0] == agent and ConstraintAstar.overlapping(edge_time, con[1]):
                    return False

        if vertex in constraints:
            for con in constraints[vertex]:
                if con[0] == agent and ConstraintAstar.overlapping(vertex_time, con[1]):
                    return False

        return True
