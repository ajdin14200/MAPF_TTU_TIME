import time
import argparse
import csv
import sys
import os
from pathlib import Path
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from pathfinding.planners.cbs_ttu import CBSTTU_Planner
from pathfinding.planners.utils.STNU import tpg_to_stnu
from pathfinding.planners.utils.tpg import build_tpg_from_solution

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
print(sys.path)

from pathfinding.planners.cbstu_list_goal import CBSTUPlanner
from pathfinding.planners.utils.tu_problem import TimeUncertaintyProblem


def print_solution(solution):
    print()
    for agent, plan in solution.paths.items():
        print("Path for agent_" + str(agent) + ": ")
        for movement in plan.path:
            print(movement[1], end="")
            if plan.path[-1] != movement:
                print(" --> ", end="")
        print()
    print("Solution cost is between " + str(solution.cost[0]) + " and " + str(solution.cost[1]))
    print("Solution length is  " + str(solution.__sizeof__()) + "\n")


def run_exp(test_map, max_agent, time_uncertainty, task_uncertainty, time_limit, data, i):

    test_map.generate_problem_instance(uncertainty=time_uncertainty)
    name_instance = "instance_"+ str(i)
    print("instance : ", i)
    test_map.generate_agents(max_agent)

    test_map.generate_agents_with_subgoals(max_agent, 3)
    test_map.generate_subgoal_action_time(uncertainty=task_uncertainty)
    test_map.fill_heuristic_table_with_goal_list()

    print("////////// CBSTU_TDU_PC //////////")

    cbsttu_planner = CBSTTU_Planner(test_map)
    #print("time_limit : ", time_limit)
    solution = cbsttu_planner.find_solution(soc=False, time_lim=time_limit, use_cat=True, use_pc=True,
                                          use_bp=True)

    data.append([name_instance, "CBSTU_TDU_PC", solution.is_solved, solution.cost[1],
                 solution.time_to_solve, solution.iteration])

    print("////////// CBSTU_TDU //////////")
    solution = cbsttu_planner.find_solution(soc=False, time_lim=time_limit, use_cat=True, use_pc=False,
                                              use_bp=True)

    data.append([name_instance, "CBSTU_TDU", solution.is_solved, solution.cost[1],
                 solution.time_to_solve, solution.iteration])


    print("////////// CBSTU_STNU_PC //////////")
    cbstu_planner = CBSTUPlanner(test_map)
    solution_cbstu_pc = cbstu_planner.find_solution(soc=False,time_lim=time_limit, use_cat=True, use_pc=True, use_bp=True)
    solved = False
    time_cbstu = solution_cbstu_pc.time_to_solve
    cost = solution_cbstu_pc.cost[1]

    if solution_cbstu_pc.is_solved:
        start = time.time()
        tpg = build_tpg_from_solution(solution_cbstu_pc)
        stnu = tpg_to_stnu(tpg, test_map.edges_and_weights, subgoal_action_time=test_map.subgoal_action_time)
        stnu.check_Strong_Controllability()
        duration = time.time() - start
        time_cbstu +=  duration
        if time_cbstu < time_limit:
            solved = True
            cost = stnu.get_cost()


    data.append([name_instance, "CBSTU_STNU_PC", solved, cost,
                 time_cbstu, solution_cbstu_pc.iteration])

    print("////////// CBSTU_STNU //////////")
    solution_cbstu = cbstu_planner.find_solution(soc=False, time_lim=time_limit, use_cat=True, use_pc=False, use_bp=True)
    solved = False
    time_cbstu = solution_cbstu.time_to_solve
    cost = solution_cbstu.cost[1]

    if solution_cbstu.is_solved:
        start = time.time()
        tpg = build_tpg_from_solution(solution_cbstu)
        stnu = tpg_to_stnu(tpg, test_map.edges_and_weights, subgoal_action_time=test_map.subgoal_action_time)
        stnu.check_Strong_Controllability()
        duration = time.time() - start
        time_cbstu += duration
        if time_cbstu < time_limit:
            solved = True
            cost = stnu.get_cost()

    data.append([name_instance, "CBSTU_STNU", solved, cost,
                 time_cbstu, solution_cbstu.iteration])



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Encoder for controllability of temporal problems')

    # Here the first two parameters are mandatory. The -- option makes the parameters optional but it is used for clarity on the input bash command
    parser.add_argument('--min_agents', metavar='min_agent', type=str, help='The minimum number of agents ')
    parser.add_argument('--max_agents', metavar='max_agent', type=str, help='The maximum number of agents ')
    parser.add_argument('--time_uncertainty', metavar='uncertainty', type=str, help='the maximum possible uncertainty')
    parser.add_argument('--task_uncertainty', metavar='uncertainty', type=str, help='the maximum possible uncertainty')
    parser.add_argument('--time', metavar='time_limit', type=str, help='the time_limit')
    parser.add_argument('--rate', metavar='rate', help='The success rate')
    parser.add_argument('--iteration', metavar='iteration', help='Number of iteration per scenario')
    parser.add_argument('--map', metavar='map', help='The map to use for experimentation')
    #can only be used with the linear_cycle option.
    # This is an additional optimization function that provides some fairness among the contracts reduction, i.e., maximize the number of contracts that can be reduced by the same amount

    args = parser.parse_args()

    if not args.max_agents or not args.min_agents or not args.time_uncertainty or not args.time or not args.iteration or not args.map:
        print("you forgot at least one parameter")

    else:

        map = os.path.join(os.path.dirname(__file__), '../../maps/benchmark/' + args.map +'.map')

        folder_name = args.map +'_'+ 'min_'+ args.min_agents + '_' + 'max_' + args.max_agents + '_TU_' + args.time_uncertainty + '_TAU_' + args.task_uncertainty
        folder_path = Path(os.path.join(os.path.dirname(__file__), '../results/' + folder_name))

        if folder_path.exists():
            shutil.rmtree(folder_path)

        folder_path.mkdir(parents=True, exist_ok=True)
        test_map = TimeUncertaintyProblem(map)

        for nb_agent in range(int(args.min_agents), int(args.max_agents) +1,2):
            print("*"*40)
            print("nb_agent : ", nb_agent)
            data = [["name", "method", "solved", "cost", "time", "nb_iter"]]

            for i in range (int(args.iteration)):
                run_exp(test_map, nb_agent, int(args.time_uncertainty), int(args.task_uncertainty), int(args.time), data, i)

            csv_file_path = f'../results/{folder_name}/A_{nb_agent}.csv'
            # Open the file in write mode
            with open(csv_file_path, mode='w', newline='') as file:
                # Create a csv.writer object
                writer = csv.writer(file)
                # Write data to the CSV file
                writer.writerows(data)

