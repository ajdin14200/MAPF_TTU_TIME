import copy

import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
print(sys.path)

from pathfinding.planners.cbs_ttu import CBSTTU_Planner
from pathfinding.planners.pibt import PIBTPlanner
from pathfinding.planners.cbstu_list_goal import CBSTUPlanner
#from pathfinding.planners.utils.STNU import *

from pathfinding.planners.utils.tu_problem import TimeUncertaintyProblem


def print_solution(solution):

    for agent, plan in solution.paths.items():
        print("Path for agent_" + str(agent) + ": ")
        for movement in plan.path:
            print(movement, end="")
            if plan.path[-1] != movement:
                print(" --> ", end="")
        print()
    print("Solution cost is between " + str(solution.cost[0]) + " and " + str(solution.cost[1]))
    print("Solution length is  " + str(solution.__sizeof__()) + "\n")



def run_test_map():
    print("----------- Small custom map ---------------")
    test_map = TimeUncertaintyProblem(os.path.join(os.path.dirname(__file__), '../../maps/benchmark/maze.map'))
    #test_map = TimeUncertaintyProblem(os.path.join(os.path.dirname(__file__), '../../maps/test_map.map'))
    TU = 1
    TAU = 1
    test_map.generate_problem_instance(uncertainty=TU)
    test_map.generate_agents_with_subgoals(3, 2)
    test_map.generate_subgoal_action_time(uncertainty=TAU)
    test_map.fill_heuristic_table_with_goal_list()

    print("////////// PIBT //////////")

    planner = PIBTPlanner(test_map)
    pibt_solution = planner.find_solution(time_limit=60, max_steps=100000, soc=False)
    print_solution(pibt_solution)

    print("////////// CBSTTU //////////")

    cbsttu_planner = CBSTTU_Planner(test_map)
    # print("time_limit : ", time_limit)
    cbsttu_solution = cbsttu_planner.find_solution(soc=False, time_lim=60, use_cat=True, use_pc=False,
                                              use_bp=True)
    print_solution(cbsttu_solution)


    print("////////// CBSTU_STNU //////////")
    ccbs_planner = CBSTUPlanner(test_map)
    cbstu_solution = ccbs_planner.find_solution(soc=False, time_lim=60, use_cat=True, use_pc=True,
                                                   use_bp=True)
    print_solution(cbstu_solution)



    # Use the following commented lines for transforming the solution into a TPG and then into an STNU.
    # To run the check_Strong_Controllability() function you need to have docplex installed to use CPLEX solver
    '''  
    tpg = build_tpg_from_solution(solution)
    tpg.print_graph()
    stnu = tpg_to_stnu(tpg, test_map.edges_and_weights, subgoal_action_time=test_map.subgoal_action_time)
    stnu.check_Strong_Controllability()
    print(stnu.get_cost())
    '''








run_test_map()
print('Done')

