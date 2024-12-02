from process_model import Machine, Operation, Job, Solution, Process
from schedule_finder import ScheduleFinder
from instances_parser import ClassicJobShopParser, FlexibleJobShopParser
import matplotlib.pyplot as plt
import time
import os
import pickle
import json
import yaml
import random
from pathlib import Path
from copy import deepcopy


def auto_test(config_path: str):
    if config_path.endswith(".json"):
        with open(config_path, 'rb') as jf:
            tests = json.load(jf)
    elif config_path.endswith(".yml"):
        with open("test_instances/classic_job_shop_test.yml", "r") as f:
            tests = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file type '{config_path.split('.')[-1]}'")

    seeds = [234, 4456, 987, 902, 1, 63494, 992815554, 22, 7830, 2310, 12345, 98765, 42, 999, 123123123, 29384756,
             76492, 62, 662341272, 77225]

    for test in tests:
        classic_mode = False
        if test["type"] == "classic":
            parser = ClassicJobShopParser()
            classic_mode = True
        elif test["type"] == "flexible":
            parser = FlexibleJobShopParser()
        else:
            raise ValuError(f'Unknown parser type: {test["type"]}')

        print(f"\n\n==================== Processing: {test['input_path']} ====================")
        process = parser.parse(test["input_path"])
        tests_out = Path(test["out_path"] + "/tests/")
        tests_out.mkdir(parents=True, exist_ok=True)

        params = test["ABC_parameters"]
        population_size = params["population_size"]
        max_iter = params["max_iter"]
        limit = params["limit"]
        local_search_max_iter = params["local_search_max_iter"]
        local_search_probability = params["local_search_probability"]
        stagnancy_terminate_limit = params["stagnancy_terminate_limit"]
        optimal_duration = params["optimal_duration"]
        try:
            acceptable_relative_error = params["acceptable_relative_error"]
        except KeyError:
            acceptable_relative_error = 0.0

        parameters = {
            'process': process,
            'population_size': population_size,
            'max_iter': max_iter,
            'limit': limit,
            'local_search_max_iter': local_search_max_iter,
            'local_search_probability': local_search_probability,
            'stagnancy_terminate_limit': stagnancy_terminate_limit,
            'optimal_duration': optimal_duration
        }

        print("\nParameters = {")
        for k, v in parameters.items():
            print(f"\t{k} = {v}")
        print(f"\tacceptable_relative_error = {acceptable_relative_error}")
        print(f"\tclassic_mode = {classic_mode}")
        print("}\n")

        with open(tests_out / f"parameters.pkl", 'wb') as f:
            pickle.dump(parameters, f)

        full_test_data = {
            "parameters": parameters,
        }

        for seed in seeds:
            print(f"\nSeed {seed}:")
            data_file = f"{tests_out}/seed_{seed}-data.pkl"

            if os.path.isfile(data_file):
                print("\tAlready exists")
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                print("\tSearching...")
                random.seed(seed)

                SF = ScheduleFinder(
                    operations=process.operations,
                    population_size=population_size,
                    max_iter=max_iter,
                    limit=limit,
                    local_search_max_iter=local_search_max_iter,
                    local_search_probability=local_search_probability,
                    stagnancy_terminate_limit=stagnancy_terminate_limit,
                    optimal_duration=optimal_duration,
                    acceptable_relative_error=acceptable_relative_error,
                    classic_mode=classic_mode
                )

                found_solution = SF.find_optimal_solution()
                process_duration = found_solution.compute_solution_duration()

                data = {
                    "seed": seed,
                    "history": SF.history,
                    "found_solution": found_solution,
                    "process_duration": process_duration,
                    "goal_function_call_counter": SF.goal_function_call_counter,
                    "relative_error": SF.relative_error
                }

                with open(tests_out / f"seed_{seed}-data.pkl", 'wb') as f:
                    pickle.dump(data, f)

            print(f"Best solution: {data['process_duration']}  (err = {data['relative_error']})")
            full_test_data[seed] = deepcopy(data)

        with open(tests_out.parent / f"data.pkl", 'wb') as f:
            pickle.dump(full_test_data, f)
