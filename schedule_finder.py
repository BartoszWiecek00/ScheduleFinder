from typing import List, Optional
from copy import deepcopy
import pandas as pd
import random as rm
import numpy as np
from process_model import Operation, Solution
import time
import matplotlib.pyplot as plt


class ScheduleFinder:
    def __init__(self,
                 operations: List[Operation],
                 population_size: int,
                 limit: int,
                 local_search_max_iter: int = 3,
                 local_search_probability: float = 0.4,
                 max_iter: int = 1000,
                 stagnancy_terminate_limit: int = 10,
                 optimal_duration: Optional[int] = None,
                 acceptable_relative_error: float = 0.0,
                 debug_msg: bool = True,
                 classic_mode: bool = False):

        self.operations: List[Operation] = deepcopy(operations)
        self.population_size: int = population_size  # == food resources number == employed bees number
        self.max_iter: int = max_iter
        self.best_solution = None
        self.best_solution_duration = float("inf")
        self.history = {}
        self.sample_schedule = [op.job_number for op in self.operations]
        self.population = {
            "schedule": [],
            "operations": [],
            "solution": [],
            "process_duration": []
        }
        self.limit = limit
        self.local_search_max_iter: int = local_search_max_iter
        self.local_search_probability: float = local_search_probability
        self.stagnancy_terminate_limit = stagnancy_terminate_limit
        self.searching_time = 0
        self.goal_function_call_counter = 0
        self.optimal_duration = optimal_duration
        self.relative_error = None if optimal_duration is None else float("inf")
        self.acceptable_relative_error = acceptable_relative_error
        self.iterations_to_best_solution = 0
        self.debug_msg = debug_msg
        self.all_permutations = self.compute_permutation_number()
        self.all_schedules_so_far = set()
        self.classic_mode = classic_mode

    def compute_permutation_number(self):
        occurrences = {}
        for num in self.sample_schedule:
            if num in occurrences:
                occurrences[num] += 1
            else:
                occurrences[num] = 1

        numerator = np.math.factorial(len(self.sample_schedule))
        denominator = 1
        for v in occurrences.values():
            denominator *= np.math.factorial(v)

        return numerator / denominator

    def init_population(self):
        while len(self.population["schedule"]) < self.population_size:
            sample_schedule = deepcopy(self.sample_schedule)
            rm.shuffle(sample_schedule)
            if sample_schedule not in self.population["schedule"]:
                self.population["schedule"].append(sample_schedule)
                operations = deepcopy(self.operations)
                for op in operations:
                    op.assign_to_random_machine()
                self.population["operations"].append(operations)
                sample_solution = Solution(operations, sample_schedule)
                self.population["solution"].append(sample_solution)
                self.population["process_duration"].append(self.compute_duration(sample_solution))

    @staticmethod
    def swap_two_elements(schedule: List[int], forbidden_indexes: List[int] = []):
        schedule_copy = deepcopy(schedule)
        possible_indexes = [ind for ind in range(len(schedule_copy)) if ind not in forbidden_indexes]
        i, j = rm.choice(possible_indexes), rm.choice(possible_indexes)
        while j == i or schedule_copy[j] == schedule_copy[i]:
            j = rm.choice(possible_indexes)

        schedule_copy[i], schedule_copy[j] = schedule_copy[j], schedule_copy[i]
        return schedule_copy, i, j

    def insert_one_element(self, schedule: List[int], forbidden_indexes: List[int] = []):
        schedule_copy = deepcopy(schedule)
        possible_indexes = [ind for ind in range(len(schedule_copy)) if ind not in forbidden_indexes]
        i, j = rm.choice(possible_indexes), rm.choice(possible_indexes)
        while j == i:
            j = rm.choice(possible_indexes)

        if i < j:
            schedule_copy.insert(j, schedule_copy[i])
            schedule_copy.pop(i)
        else:
            elem = schedule_copy.pop(i)
            schedule_copy.insert(j, elem)

        if schedule_copy == schedule:
            return self.insert_one_element(schedule)

        return schedule_copy, i, j

    def double_swap(self, schedule: List[int]):
        result, ind0, ind1 = self.swap_two_elements(schedule)
        result, _, _ = self.swap_two_elements(result, [ind0, ind1])
        return result

    def double_insert(self, schedule: List[int]):
        result, ind0, ind1 = self.insert_one_element(schedule)
        result, _, _ = self.insert_one_element(result, [ind0, ind1])
        return result

    def single_swap(self, schedule: List[int]):
        result, _, _ = self.swap_two_elements(schedule)
        return result

    def single_insert(self, schedule: List[int]):
        result, _, _ = self.insert_one_element(schedule)
        return result

    def choose_searching_method(self, forbidden_methods=[]):
        all_methods = [self.single_swap, self.single_insert, self.double_swap, self.double_insert]
        return rm.choice([method for method in all_methods if method not in forbidden_methods])

    def choose_single_searching(self, forbidden_methods=[]):
        all_methods = [self.single_swap, self.single_insert]
        return rm.choice([method for method in all_methods if method not in forbidden_methods])

    def randomly_change_machine_assignment(self, operations: List[Operation]):
        if self.classic_mode:
            return deepcopy(operations)

        operations_copy = deepcopy(operations)
        operations_with_more_than_one_workable_machine = []
        for op in operations_copy:
            if len(op.workable_at_machines) > 1:
                operations_with_more_than_one_workable_machine.append(op)

        if not operations_with_more_than_one_workable_machine:  # Happens for Classic Jop Scheduling Problem
            return operations_copy

        op = rm.choice(operations_with_more_than_one_workable_machine)
        new_machine = rm.choice(op.workable_at_machines)
        while new_machine == op.machine_ID:
            new_machine = rm.choice(op.workable_at_machines)

        op.assign_to_machine(new_machine)

        return operations_copy

    def choose_reassignment_method(self, forbidden_methods=[]):
        all_methods = [self.randomly_change_machine_assignment]
        return rm.choice([method for method in all_methods if method not in forbidden_methods])

    def generate_new_solution(self, population_ind: int, search_method, reassignment_method):
        actual_schedule = self.population["schedule"][population_ind]
        actual_operations = self.population["operations"][population_ind]

        new_schedule = search_method(actual_schedule)
        new_operations = reassignment_method(actual_operations)
        new_solution = Solution(new_operations, new_schedule)

        return new_solution

    @staticmethod
    def rank_selection(solutions, variant=1, power=2):
        solutions_rank = sorted(solutions, key=lambda x: x[0])
        s_num = len(solutions_rank)

        if variant == 1:
            indexes = [0 for _ in range(s_num)]
            for i in range(s_num):
                for _ in range((s_num - i) ** power):
                    indexes[i] += rm.randint(0, 100)
            idx = indexes.index(max(indexes))
        elif variant == 2:
            indexes = []
            for i in range(s_num):
                indexes.extend([i for _ in range((s_num - i) ** power)])
            idx = rm.choice(indexes)
        elif variant == 3:
            return solutions_rank[0]
        else:
            raise ValueError(f"Unknown rank selection variant '{variant}'")

        return solutions_rank[idx]

    @staticmethod
    def best_selection(solutions):
        solutions_rank = sorted(solutions, key=lambda x: x[0])
        return solutions_rank[0]

    def compute_duration(self, solution):
        self.goal_function_call_counter += 1
        return solution.compute_solution_duration()

    def update_population(self, ind, duration, solution):
        self.population["schedule"][ind] = solution.schedule
        self.population["solution"][ind] = solution
        self.population["operations"][ind] = solution.operations
        self.population["process_duration"][ind] = duration

    def employed_bees_phase(self):
        for eb_ind in range(self.population_size):
            solutions = [(self.population["process_duration"][eb_ind], self.population["solution"][eb_ind])]

            search_method = self.choose_searching_method()
            new_solution = self.generate_new_solution(eb_ind, search_method, self.choose_reassignment_method())
            solutions.append((self.compute_duration(new_solution), new_solution))

            # Local search
            if rm.uniform(0, 1) < self.local_search_probability:
                local_search_method = self.single_swap \
                    if search_method in [self.single_insert, self.double_insert] else self.single_insert
                for _ in range(rm.randint(1, self.local_search_max_iter)):
                    new_solution = self.generate_new_solution(eb_ind, local_search_method, self.choose_reassignment_method())
                    solutions.append((self.compute_duration(new_solution), new_solution))

            chosen_solution_duration, chosen_solution = self.rank_selection(solutions)
            self.update_population(eb_ind, chosen_solution_duration, chosen_solution)

    def onlooker_phase(self):
        for ob_ind in range(self.population_size):
            # Select solution for tournament
            other_ind = rm.choice([i for i in range(self.population_size) if i != ob_ind])
            if self.population["solution"][other_ind].better_than(self.population["solution"][ob_ind]):
                winner_ind = other_ind
            else:
                winner_ind = ob_ind

            new_solution = self.generate_new_solution(winner_ind, self.choose_searching_method(),
                                                      self.choose_reassignment_method())
            new_solution_duration = self.compute_duration(new_solution)
            solutions = [
                (self.population["process_duration"][winner_ind], self.population["solution"][winner_ind]),
                (new_solution_duration, new_solution)
            ]
            chosen_solution_duration, chosen_solution = self.best_selection(solutions)
            self.update_population(winner_ind, chosen_solution_duration, chosen_solution)

    def update_history(self, epoch):
        for i in range(self.population_size):
            self.history[epoch]["population_epoch_end"][i] = deepcopy(self.population["solution"][i])
            self.history[epoch]["duration"][i] = self.population["process_duration"][i]
            new_better = self.history[epoch]["population_epoch_end"][i].better_than(
                self.history[epoch]["population_epoch_start"][i])
            self.history[epoch]["solution_improved"][i] = new_better
            if new_better:
                self.history[epoch]["chances_for_improvement_left"][i] = self.limit
            else:
                self.history[epoch]["chances_for_improvement_left"][i] -= 1

            if self.population["process_duration"][i] < self.history[epoch]["best_in_epoch_duration"]:
                self.history[epoch]["best_in_epoch_duration"] = self.population["process_duration"][i]
                self.history[epoch]["best_in_epoch"] = deepcopy(self.population["solution"][i])

    def scout_phase(self, epoch, base_on_best_solution=False):
        solution_to_replace_ids = []
        for i in range(self.population_size):
            if self.history[epoch]["chances_for_improvement_left"][i] <= 0:
                solution_to_replace_ids.append(i)

        if not solution_to_replace_ids:
            return

        for replace_id in solution_to_replace_ids:
            if self.best_solution is None or self.history[epoch]["best_in_epoch"].better_than(self.best_solution):
                best_solution = deepcopy(self.history[epoch]["best_in_epoch"])
            else:
                best_solution = deepcopy(self.best_solution)

            if base_on_best_solution:
                new_schedule = self.single_insert(best_solution.get_schedule())
                for _ in range(rm.randint(0, 3)):
                    new_schedule = self.single_insert(new_schedule)
            else:
                new_schedule = deepcopy(self.sample_schedule)
                rm.shuffle(new_schedule)
                if len(self.all_schedules_so_far) < self.all_permutations:
                    while tuple(new_schedule) in self.all_schedules_so_far:
                        rm.shuffle(new_schedule)

            self.all_schedules_so_far.add(tuple(new_schedule))
            new_solution = Solution(self.population["operations"][replace_id], new_schedule)
            new_solution_duration = self.compute_duration(new_solution)
            self.update_population(replace_id, new_solution_duration, new_solution)

            self.history[epoch]["population_epoch_end"][replace_id] = deepcopy(new_solution)
            self.history[epoch]["chances_for_improvement_left"][replace_id] = self.limit
            self.history[epoch]["solution_improved"][replace_id] = new_solution.better_than(
                self.history[epoch]["population_epoch_start"][replace_id])

            if new_solution.better_than(best_solution):
                self.history[epoch]["best_in_epoch"] = deepcopy(new_solution)
                self.history[epoch]["best_in_epoch_duration"] = self.population["process_duration"][replace_id]

    def remove_unnecessary_history(self, epoch):
        if epoch not in self.history:
            return
        for key in ["population_epoch_start", "population_epoch_end", "solution_improved",
                    "chances_for_improvement_left", "duration", "best_in_epoch", "worst_in_epoch"]:
            del self.history[epoch][key]

    def memorize_solution_statistics(self, epoch):
        worst_ind = self.history[epoch]["duration"].index(max(self.history[epoch]["duration"]))
        self.history[epoch]["worst_in_epoch"] = deepcopy(self.history[epoch]["population_epoch_end"][worst_ind])
        self.history[epoch]["worst_in_epoch_duration"] = self.history[epoch]["duration"][worst_ind]
        self.history[epoch]["mean_duration_in_epoch"] = np.mean(self.history[epoch]["duration"])
        self.history[epoch]["std_duration_in_epoch"] = np.std(self.history[epoch]["duration"])

        if self.best_solution is None or self.history[epoch]["best_in_epoch"].better_than(self.best_solution):
            self.best_solution = deepcopy(self.history[epoch]["best_in_epoch"])
            self.best_solution_duration = self.history[epoch]["best_in_epoch_duration"]
            self.iterations_to_best_solution = epoch + 1

            if self.relative_error is not None:
                self.relative_error = (self.best_solution_duration - self.optimal_duration) / self.optimal_duration

        self.history[epoch]["relative_error"] = self.relative_error

        if epoch >= 2:
            self.remove_unnecessary_history(epoch - 2)

    def init_history_for_epoch(self, epoch):
        if epoch == 0:
            chances_for_improvement_left = [self.limit for _ in range(self.population_size)]
        else:
            chances_for_improvement_left = [limit for limit in self.history[epoch - 1]["chances_for_improvement_left"]]

        for sol in self.population["solution"]:
            self.all_schedules_so_far.add(tuple(sol.get_schedule()))

        self.history[epoch] = {
            "population_epoch_start": [deepcopy(solution) for solution in self.population["solution"]],
            "population_epoch_end": [None for _ in range(self.population_size)],
            "solution_improved": [None for _ in range(self.population_size)],
            "chances_for_improvement_left": chances_for_improvement_left,
            "duration": [float("inf") for _ in range(self.population_size)],
            "best_in_epoch": None,
            "best_in_epoch_duration": float("inf"),
            "worst_in_epoch": None,
            "worst_in_epoch_duration": float("inf"),
            "mean_duration_in_epoch": float("inf"),
            "std_duration_in_epoch": float("inf"),
            "relative_error": None
        }

    def stop_condition_meet(self, stagnancy_iteration_count, best_solution_before):
        if self.optimal_duration is not None and self.best_solution_duration <= self.optimal_duration:
            return True, stagnancy_iteration_count

        if self.relative_error is not None and self.relative_error <= self.acceptable_relative_error:
            return True, stagnancy_iteration_count

        if best_solution_before is None or self.best_solution.better_than(best_solution_before):
            stagnancy_iteration_count = 0
        else:
            stagnancy_iteration_count += 1

        return stagnancy_iteration_count >= self.stagnancy_terminate_limit, stagnancy_iteration_count

    def find_optimal_solution(self, return_history: bool = False):
        # Initialize parameters - done in __init__

        start = time.time()

        # Initialize population
        self.init_population()

        stagnancy_iteration_count = 0
        for iteration_nr in range(self.max_iter):
            iteration_start = time.time()
            if self.debug_msg:
                print(f"Iteration nr: {iteration_nr}. Best solution so far takes: {self.best_solution_duration}")

            # Remember best solution at the beginning of iteration
            best_solution_before = deepcopy(self.best_solution)

            self.init_history_for_epoch(iteration_nr)

            # Employed bee phase
            self.employed_bees_phase()

            # Onlooker phase
            self.onlooker_phase()

            # Update history
            self.update_history(iteration_nr)

            # Scout phase
            self.scout_phase(iteration_nr)

            # Memorize the best solution achieved so far with some additional statistics.
            self.memorize_solution_statistics(iteration_nr)

            print(f"\tIteration took {time.time() - iteration_start} [sec]")
            print(f"\tRelative error = {self.history[iteration_nr]['relative_error']}")

            # Check if terminate
            terminate, stagnancy_iteration_count = self.stop_condition_meet(stagnancy_iteration_count, best_solution_before)
            if terminate:
                self.remove_unnecessary_history(iteration_nr - 1)
                self.remove_unnecessary_history(iteration_nr)
                break

        self.searching_time = time.time() - start

        print(f"\tSearching took {self.searching_time} [sec]")
        print(f"\tBest solution relative error =  {self.relative_error}")
        if return_history:
            return deepcopy(self.best_solution), deepcopy(self.history)

        return deepcopy(self.best_solution)

    def make_chart_for_history(self):
        if not self.history:
            return None

        epoch = []
        best_solution_duration = []
        worst_solution_duration = []
        mean_solution_duration = []
        std_solution_duration = []
        relative_error = []

        for i in range(len(self.history)):
            epoch.append(i)
            best_solution_duration.append(self.history[i]["best_in_epoch_duration"])
            worst_solution_duration.append(self.history[i]["worst_in_epoch_duration"])
            mean_solution_duration.append(self.history[i]["mean_duration_in_epoch"])
            std_solution_duration.append(self.history[i]["std_duration_in_epoch"])
            relative_error.append(self.history[i]["relative_error"])

        y_data = [
            best_solution_duration,
            worst_solution_duration,
            mean_solution_duration,
            std_solution_duration
        ]

        y_labels = [
            "Best solution",
            "Worst solution",
            "Mean solution",
            "Std solution",
        ]

        if relative_error[0] is not None:
            y_data.append([r_err * 100 for r_err in relative_error])
            y_labels.append("Relative error [%]")

        if len(epoch) <= 50:
            xticks_step = 1
        elif 50 < len(epoch) <= 100:
            xticks_step = 2
        elif 100 < len(epoch) <= 150:
            xticks_step = 5
        elif 150 < len(epoch) <= 200:
            xticks_step = 10
        elif 200 < len(epoch) <= 300:
            xticks_step = 15
        else:
            xticks_step = 20

        xticks = [epoch[i] for i in range(0, len(epoch), xticks_step)]

        fig, axs = plt.subplots(len(y_data), 1, sharex=True, figsize=(10, 5))
        fig.subplots_adjust(hspace=0)

        for i in range(len(y_data)):
            if i == 0 and self.optimal_duration is not None:
                axs[i].plot(epoch, y_data[i], label="Computed")
                axs[i].plot(epoch, [self.optimal_duration] * len(epoch), label="Optimal")
                axs[i].legend()
            else:
                axs[i].plot(epoch, y_data[i])

            axs[i].set_xticks(xticks)
            axs[i].set_ylabel(y_labels[i])
            axs[i].grid(linewidth=0.5, alpha=0.5)

        axs[-1].set_xlabel("Iterations")

        fig.suptitle(f"Statistics over algorithm iterations (goal function calls =  {self.goal_function_call_counter})",
                     y=0.92)

        return fig
