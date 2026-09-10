from docplex.mp.model import Model
import networkx as nx
import matplotlib.pyplot as plt



WAIT_TIME = 1
INF = 10**9

class STNU:
    def __init__(self):
        self.timepoints = set()
        self.requirement_constraints = []   # (source, target, lb, ub)
        self.contingent_constraints = []    # (source, target, lb, ub)
        self.model = Model()
        self.variables = {}
        self.solved = False

    def __get_uncontrollable_timepoints_(self):

        uncontrollable_timepoints = set()
        for constraint in self.contingent_constraints:
            uncontrollable_timepoints.add(constraint[1])

        return uncontrollable_timepoints

    def __get_controllable_timepoints(self, uncontrollables):

        for timepoint in self.timepoints:
            if timepoint not in uncontrollables:
                self.variables[timepoint] = self.model.continuous_var(name=timepoint)

    def __get_variables_parent(self):
        parents_map = {}
        parents_contingent = {}
        for constraint in self.contingent_constraints:

            source = constraint[0]
            dest = constraint[1]
            parents_map[dest] = source
            parents_contingent[dest] = constraint
        return parents_map, parents_contingent

    # there is no need for a while loop here as chain of contingents do not exist here.
    def __get_chain_formulas(self, parents_map, parents_contingent):

        chain_formulas_map = {}

        for tp, father in parents_map.items():
            z = father
            constraint = parents_contingent[tp]
            l = constraint[2]
            u = constraint[3]

            chain_l = l
            chain_u = u

            while z not in self.variables:
                constraint = parents_contingent[z]
                chain_l += constraint[2]
                chain_u += constraint[3]
                z = parents_map[z]

            chain_l += self.variables[z]
            chain_u += self.variables[z]
            chain_formulas_map[tp] = (chain_l, chain_u)

        return chain_formulas_map

    def __encoding_requirements(self, chain_formulas_map):

        constraints = [self.variables["S"] == 0]
        for requirement in self.requirement_constraints:


            source = requirement[0]
            dest = requirement[1]
            l = requirement[2]
            u = requirement[3]

            vs = None
            vd = None

            # Here we test the controllability of each timepoints.
            # Note that we didn't update all the constraint in the owned_contract_as_contingent() function as we can bypass this using the set of timepoints
            if source in self.variables and dest in self.variables:

                vs = self.variables[source]
                vd = self.variables[dest]
                constraints.append(vd - vs >= l)
                constraints.append(vd - vs <= u)

            elif source in self.variables and dest in chain_formulas_map: # source controllable and dest uncontrollable
                vs = self.variables[source]
                vd = chain_formulas_map[dest]
                constraints.append(vd[0] - vs >= l)
                constraints.append(vd[1] - vs <= u)
                # print("case 1: ",vd[0] - vs >= l, " and ", vd[1] - vs <= u )

            elif source in chain_formulas_map and dest in self.variables: # source uncontrollable and dest controllable
                vs = chain_formulas_map[source]
                vd = self.variables[dest]
                constraints.append(vd - vs[1] >= l)
                constraints.append(vd - vs[0] <= u)

            else: #source and dest uncontrollable
                vs = chain_formulas_map[source]
                vd = chain_formulas_map[dest]
                constraints.append(vd[0] - vs[1] >= l)
                constraints.append(vd[1] - vs[0] <= u)

        return constraints

    def print_SC_solution(self):
        print("\n=== SOLUTION ===")
        for name, variable in self.variables.items():
            print("Variable : ",name, " has value : ", variable.solution_value, "\n")
        print("\n=== COST ===")
        print(self.variables["F"].solution_value)

    def get_cost(self):

        return self.variables["F"].solution_value


    def check_Strong_Controllability(self, time_limit = 300):

        uncontrollable_timepoints = self.__get_uncontrollable_timepoints_()
        self.__get_controllable_timepoints(uncontrollable_timepoints)
        parents_map, parent_contingent = self.__get_variables_parent()  # get parent for chain of contingents
        chain_formulas_map = self.__get_chain_formulas(parents_map, parent_contingent)
        constraints = self.__encoding_requirements(chain_formulas_map)
        #print(constraints)
        self.model.add_constraints(constraints)
        self.model.minimize(self.variables["F"])
        self.model.set_time_limit(time_limit)
        result = self.model.solve()

        if result is None:
            self.solved = False
            print("No feasible solution found")
        else:
            self.solved = True
            print("A solution exist with cost : ", self.get_cost())
            print(result)

    def add_timepoint(self, tp):
        self.timepoints.add(tp)

    def add_requirement(self, source, target, lb, ub):
        self.requirement_constraints.append((source, target, lb, ub))

    def add_contingent(self, source, target, lb, ub):
        self.contingent_constraints.append((source, target, lb, ub))

    def print_stnu(self):
        print("\n=== TIMEPOINTS ===")
        for tp in sorted(self.timepoints):
            print(tp)

        print("\n=== REQUIREMENT CONSTRAINTS ===")
        for c in self.requirement_constraints:
            print(c)

        print("\n=== CONTINGENT CONSTRAINTS ===")
        for c in self.contingent_constraints:
            print(c)

    def visualize_stnu_ordered(self, tpg):
        G = nx.DiGraph()

        # Add nodes
        for tp in self.timepoints:
            G.add_node(tp)

        # Add edges
        for (src, dst, lb, ub) in self.requirement_constraints:
            G.add_edge(src, dst, label=f"[{lb},{ub}]", style="solid")

        for (src, dst, lb, ub) in self.contingent_constraints:
            G.add_edge(src, dst, label=f"[{lb},{ub}]", style="dashed")

        # -----------------------------
        # Build positions manually
        # -----------------------------
        pos = {}

        x_gap = 3
        y_gap = 3

        # S and F
        pos["S"] = (0, 0)
        pos["F"] = (10 * x_gap, 0)

        agent_nodes = get_agent_nodes_in_order(tpg)

        for agent, nodes in agent_nodes.items():
            y = -agent * y_gap  # each agent on its own row

            for i, (node_id, _) in enumerate(nodes):
                x = (i + 1) * x_gap

                # Place arrival/task-start/task-end/leaving timepoints separately.
                # Task nodes follow the paper's A -> TS -> TE -> L structure.
                has_task = (
                    f"{node_id}_TS" in self.timepoints
                    or f"{node_id}_TE" in self.timepoints
                )

                if f"{node_id}_A" in self.timepoints:
                    pos[f"{node_id}_A"] = (x, y + (0.75 if has_task else 0.5))

                if f"{node_id}_TS" in self.timepoints:
                    pos[f"{node_id}_TS"] = (x, y + 0.25)

                if f"{node_id}_TE" in self.timepoints:
                    pos[f"{node_id}_TE"] = (x, y - 0.25)

                if f"{node_id}_L" in self.timepoints:
                    pos[f"{node_id}_L"] = (x, y - (0.75 if has_task else 0.5))

        # -----------------------------
        # Draw graph
        # -----------------------------
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d["style"] == "solid"]
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d["style"] == "dashed"]

        plt.figure(figsize=(14, 6))

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=8)

        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2)
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, style="dashed", width=2)

        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title("STNU (Ordered by Path)")
        plt.axis("off")
        plt.show()







def get_edge_weight(edges_and_weights, u, v):
    """
    edges_and_weights is the structure from TimeUncertaintyProblem:
        edges_and_weights[u] = [(neighbor, (lb, ub)), ...]
    """
    if u not in edges_and_weights:
        raise ValueError(f"Vertex {u} not found in edges_and_weights")

    for neigh, interval in edges_and_weights[u]:
        if neigh == v:
            return interval

    raise ValueError(f"Missing edge weight for transition {u} -> {v}")


def arrival_tp(node_id):
    return f"{node_id}_A"


def task_start_tp(node_id):
    return f"{node_id}_TS"


def task_end_tp(node_id):
    return f"{node_id}_TE"


def task_tp(node_id):
    """Backward-compatible alias for the task-end timepoint.

    The original implementation used a single ``_T`` timepoint.  The fixed
    implementation follows the paper and uses distinct ``_TS`` (controllable
    task start) and ``_TE`` (uncontrollable task end) timepoints.
    """
    return task_end_tp(node_id)


def leaving_tp(node_id):
    return f"{node_id}_L"

def get_agent_nodes_in_order(tpg):
    """
    Returns:
        dict agent -> list of (node_id, node)
    ordered by node.index
    """
    per_agent = {}

    for node_id, node in tpg.nodes.items():
        per_agent.setdefault(node.agent, []).append((node_id, node))

    for agent in per_agent:
        per_agent[agent].sort(key=lambda x: x[1].index)

    return per_agent

def _get_agent_task_duration(subgoal_action_time, agent, location, agent_subgoals=None):
    """Return the task-duration interval for ``agent`` at ``location``.

    Two explicit, agent-aware input formats are supported:

    1. Preferred combined format::

           subgoal_action_time = {
               agent_id: {
                   location: (lb, ub),
                   ...
               },
               ...
           }

       In this form the presence of ``location`` in the per-agent mapping also
       encodes ownership of the task.

    2. Paper-style split format::

           subgoal_action_time = {location: (lb, ub), ...}
           agent_subgoals = {agent_id: {location, ...}, ...}

       This mirrors the paper's global task-duration function psi(v) together
       with the agent-specific intermediate-goal set M_a.

    The old global ``{location: (lb, ub)}`` format *without* ``agent_subgoals``
    is deliberately rejected: by itself it cannot tell whether an agent owns a
    task or is merely passing through the same location.
    """
    if not subgoal_action_time:
        return None

    if agent_subgoals is not None:
        owned_locations = agent_subgoals.get(agent, ())
        if location not in owned_locations:
            return None
        if location not in subgoal_action_time:
            raise ValueError(
                f"Agent {agent} owns a task at {location}, but no duration "
                "is present in subgoal_action_time."
            )
        return subgoal_action_time[location]

    per_agent = subgoal_action_time.get(agent)
    if per_agent is None:
        return None

    if not isinstance(per_agent, dict):
        raise ValueError(
            "Agent-specific task ownership is required. Pass "
            "subgoal_action_time as {agent: {location: (lb, ub)}} or pass "
            "agent_subgoals={agent: {location, ...}} together with the "
            "paper-style global {location: (lb, ub)} duration mapping."
        )

    return per_agent.get(location)


def tpg_to_stnu(tpg, edges_and_weights, subgoal_action_time=None, agent_subgoals=None):
    """Convert a TPG into an STNU using the construction from the paper.

    Node semantics
    --------------
    Normal node N::

        N_A --[0, INF] requirement--> N_L

    Task/subgoal node N::

        N_A  --[0, INF] requirement--> N_TS
        N_TS --[lb, ub] contingent---> N_TE
        N_TE --[0, INF] requirement--> N_L

    Thus ``N_TS`` is controllable and ``N_TE`` is uncontrollable.  This is the
    key difference from the original implementation, which attached the task
    contingent link directly to the arrival timepoint.

    Task ownership
    --------------
    ``subgoal_action_time`` should preferably be agent-specific::

        {
            agent_id: {location: (lb, ub), ...},
            ...
        }

    Alternatively, to mirror the notation in the paper, use a global duration
    mapping plus explicit agent ownership::

        subgoal_action_time = {location: (lb, ub), ...}
        agent_subgoals = {agent_id: {location, ...}, ...}

    A task is performed only on the first visit to a location assigned to that
    agent.  An agent that merely passes through another agent's task location
    gets a normal arrival-to-leaving requirement instead.

    TPG edge semantics
    ------------------
    Type-1 edge:
        source_L -> target_A, with the traversal interval.  Deterministic
        intervals are requirements; uncertain intervals are contingent links.

    Type-2 edge:
        source_L -> target_A with requirement [1, INF].
    """

    stnu = STNU()

    if subgoal_action_time is None:
        subgoal_action_time = {}

    # Without an explicit agent_subgoals mapping, the task-duration mapping
    # itself must encode ownership as {agent: {location: (lb, ub)}}.  Reject
    # the old global {location: (lb, ub)} form instead of silently assigning
    # tasks to every agent that happens to visit the location.
    if subgoal_action_time and agent_subgoals is None:
        if not all(isinstance(per_agent, dict) for per_agent in subgoal_action_time.values()):
            raise ValueError(
                "Agent-specific task ownership is required. Pass "
                "subgoal_action_time as {agent: {location: (lb, ub)}} or pass "
                "agent_subgoals={agent: {location, ...}} together with the "
                "paper-style global {location: (lb, ub)} duration mapping."
            )

    stnu.add_timepoint("S")
    stnu.add_timepoint("F")

    agent_nodes = get_agent_nodes_in_order(tpg)

    # --------------------------------------------------
    # 1) Create node timepoints
    # --------------------------------------------------
    performed_subgoal_action = {
        agent: set() for agent in agent_nodes
    }

    for agent, ordered_nodes in agent_nodes.items():
        for node_id, node in ordered_nodes:

            n_a = arrival_tp(node_id)
            n_l = leaving_tp(node_id)

            stnu.add_timepoint(n_a)
            stnu.add_timepoint(n_l)

            task_duration = _get_agent_task_duration(
                subgoal_action_time,
                agent,
                node.location,
                agent_subgoals=agent_subgoals,
            )

            is_subgoal_task = (
                task_duration is not None
                and node.location not in performed_subgoal_action[agent]
            )

            if is_subgoal_task:
                n_ts = task_start_tp(node_id)
                n_te = task_end_tp(node_id)
                stnu.add_timepoint(n_ts)
                stnu.add_timepoint(n_te)

                lb, ub = task_duration

                # The agent may wait after arrival before starting its task.
                stnu.add_requirement(n_a, n_ts, 0, INF)

                # Task duration is uncertain: controllable start ->
                # uncontrollable end, exactly as in the paper.
                stnu.add_contingent(n_ts, n_te, lb, ub)

                # The agent may leave any time after the task is finished.
                stnu.add_requirement(n_te, n_l, 0, INF)

                # A task is executed only once, on the first visit.
                performed_subgoal_action[agent].add(node.location)

            else:
                # Normal node: can leave any time after arrival.
                stnu.add_requirement(n_a, n_l, 0, INF)

    # --------------------------------------------------
    # 2) Connect S and F
    # --------------------------------------------------
    for agent, ordered_nodes in agent_nodes.items():
        first_node_id = ordered_nodes[0][0]
        last_node_id = ordered_nodes[-1][0]

        stnu.add_requirement("S", arrival_tp(first_node_id), 0, 0)
        stnu.add_requirement(leaving_tp(last_node_id), "F", 0, INF)

    # --------------------------------------------------
    # 3) Transform TPG edges
    # --------------------------------------------------
    for edge in tpg.edges:
        source_id = edge.source
        target_id = edge.target

        source_node = tpg.nodes[source_id]
        target_node = tpg.nodes[target_id]

        source_l = leaving_tp(source_id)
        target_a = arrival_tp(target_id)

        # -----------------------------
        # Type-1 edge: same-agent order
        # -----------------------------
        if edge.edge_type == 1:

            if source_node.location == target_node.location:
                # Explicit wait edge
                lb, ub = WAIT_TIME, WAIT_TIME
            else:
                lb, ub = get_edge_weight(
                    edges_and_weights,
                    source_node.location,
                    target_node.location
                )

            if lb == ub:
                stnu.add_requirement(source_l, target_a, lb, ub)
            else:
                stnu.add_contingent(source_l, target_a, lb, ub)

        # -----------------------------
        # Type-2 edge: inter-agent priority
        # -----------------------------
        elif edge.edge_type == 2:
            stnu.add_requirement(source_l, target_a, 1, INF)

    return stnu
