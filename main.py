from process_model import Machine, Operation, Job, Solution, Process
from schedule_finder import ScheduleFinder
from instances_parser import JobShopParser
import matplotlib.pyplot as plt
import time
import os
import pickle
from auto_tester import auto_test
import pickle
import numpy as np


def print_solution(solution, process, process_duration):
    print("Solution = \n{")
    sol = []
    for k, v in solution.items():
        sol.append((k, v))

    for job in process.jobs:
        line = "\t"
        for op in job.operations:
            pop_idx = -1
            for idx, s in enumerate(sol):
                if s[0].split(" ")[0] == op.ID:
                    pop_idx = idx
                    break

            s = sol.pop(pop_idx)
            line += f"{s[0]}: {s[1]}, "
        print(line[:-2])
    print("}\n")
    print(f"Total duartion = {process_duration}")


def evaluate_auto_test_output(data_path: str):
    if data_path.endswith(".pkl"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = {}
        for pkl in os.listdir(data_path):
            if pkl == "parameters.pkl":
                with open(data_path + '/' + pkl, "rb") as f:
                    data["parameters"] = pickle.load(f)
                continue

            seed = int(pkl[5:].split("-")[0])
            with open(data_path + '/' + pkl, "rb") as f:
                data[seed] = pickle.load(f)

    durations = []
    goal_fcn_calls = []
    solutions = []
    relative_errors = []
    histories = []

    for k, v in data.items():
        if k == "parameters":
            continue

        durations.append(v["process_duration"])
        goal_fcn_calls.append(v["goal_function_call_counter"])
        solutions.append(v["found_solution"])
        relative_errors.append(v["relative_error"])
        histories.append(v["history"])

    best_solution_ind = 0
    worst_solution_ind = 0
    for i in range(1, len(durations)):
        if durations[i] < durations[best_solution_ind] or (
                durations[i] == durations[best_solution_ind] and goal_fcn_calls[i] < goal_fcn_calls[best_solution_ind]):
            best_solution_ind = i
        if durations[i] > durations[worst_solution_ind] or (
                durations[i] == durations[worst_solution_ind] and goal_fcn_calls[i] > goal_fcn_calls[worst_solution_ind]):
            worst_solution_ind = i

    print("\n\nParameters:")
    for param_name, param_value in data["parameters"].items():
        print(f"\t{param_name}: {param_value}")
    print()

    print("\n\nBEST SOLUTION\n========================================")
    solutions[best_solution_ind].visualize()
    print_solution(solutions[best_solution_ind].get_solution(), data["parameters"]["process"], durations[best_solution_ind])

    SF = ScheduleFinder(
        operations=solutions[best_solution_ind].operations,
        population_size=data["parameters"]["population_size"],
        max_iter=data["parameters"]["max_iter"],
        limit=data["parameters"]["limit"],
        local_search_max_iter=data["parameters"]["local_search_max_iter"],
        local_search_probability=data["parameters"]["local_search_probability"],
        stagnancy_terminate_limit=data["parameters"]["stagnancy_terminate_limit"],
        optimal_duration=data["parameters"]["optimal_duration"]
    )

    SF.history = histories[best_solution_ind]
    SF.goal_function_call_counter = goal_fcn_calls[best_solution_ind]
    SF.make_chart_for_history()
    print(f"Relative error = {relative_errors[best_solution_ind]}")
    plt.show()
    print("========================================\n\nWORST SOLUTION\n========================================")
    solutions[worst_solution_ind].visualize()
    print_solution(solutions[worst_solution_ind].get_solution(), data["parameters"]["process"],
                   durations[worst_solution_ind])

    SF = ScheduleFinder(
        operations=solutions[worst_solution_ind].operations,
        population_size=data["parameters"]["population_size"],
        max_iter=data["parameters"]["max_iter"],
        limit=data["parameters"]["limit"],
        local_search_max_iter=data["parameters"]["local_search_max_iter"],
        local_search_probability=data["parameters"]["local_search_probability"],
        stagnancy_terminate_limit=data["parameters"]["stagnancy_terminate_limit"],
        optimal_duration=data["parameters"]["optimal_duration"]
    )

    SF.history = histories[worst_solution_ind]
    SF.goal_function_call_counter = goal_fcn_calls[worst_solution_ind]
    SF.make_chart_for_history()
    print(f"Relative error = {relative_errors[worst_solution_ind]}")
    plt.show()
    print("\n\n========================================")
    mean_duration = np.mean(durations)
    mean_goal_fcn_calls = np.mean(goal_fcn_calls)
    mean_relative_errors = np.mean(relative_errors)

    std_duration = np.std(durations)
    std_goal_fcn_calls = np.std(goal_fcn_calls)
    std_relative_errors = np.std(relative_errors)

    print("\n\nMean results")
    print(f"\tTotal duration: {mean_duration}  (std: {std_duration})")
    print(f"\tGoal function calls: {mean_goal_fcn_calls}  (std: {std_goal_fcn_calls})")
    print(f"\tRelative error: {mean_relative_errors}  (std: {std_relative_errors})")

    print(f"\nAll relative errors: {relative_errors}\n")


def main():
    auto_test("test_instances/classic_job_shop_test.yml")

    for p in ["test_out/JSP/la01/data.pkl", "test_out/JSP/orb09/data.pkl", "test_out/JSP/la26/data.pkl",
              "test_out/FJSP/Kacem1/data.pkl", "test_out/FJSP/car7-vdata/data.pkl", "test_out/FJSP/la16-rdata/data.pkl"]:
        print(f"\n\n------------------ {p} ------------------")
        evaluate_auto_test_output(p)


if __name__ == "__main__":
    main()
