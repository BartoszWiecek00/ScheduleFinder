from typing import List, Optional, Dict
import operator
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as rm


class Machine:
    def __init__(self, ID):
        self.ID = ID
        self.remaining_occupancy_time = 0


class Operation:
    def __init__(self, ID: str, duration_tab: Dict[str, List[int]], machine_ID: Optional[str] = None):
        self.ID = ID
        self.duration_tab = deepcopy(duration_tab)
        self.workable_at_machines = [m for m in self.duration_tab.keys()]
        self.job_number = int(ID[1:].split("_")[0])
        self.job_ID = f"J{self.job_number}"
        self.number = int(ID.split("_")[1])

        self.machine_ID = machine_ID
        if self.machine_ID is not None:
            self.duration = self.duration_tab[machine_ID]
        else:
            self.duration = None

    def __str__(self):
        if self.machine_ID is None:
            return f"{self.ID}{{job_ID={self.job_ID}, duration=UNKNOWN, machine=UNASSIGNED}}"

        return f"{self.ID}{{job_ID={self.job_ID}, duration={self.duration_tab[self.machine_ID]}, machine={self.machine_ID}}}"

    def assign_to_machine(self, machine_ID):
        self.machine_ID = machine_ID
        self.duration = self.duration_tab[machine_ID]

    def assign_to_random_machine(self):
        self.machine_ID = rm.choice(self.workable_at_machines)
        self.duration = self.duration_tab[self.machine_ID]


class Job:
    def __init__(self, ID: str, operations: List[Operation]):
        self.ID = ID
        self.operations = operations


class Process:
    def __init__(self, machines: List[Machine], operations: List[Operation], jobs: List[Job]):
        self.machines = deepcopy(machines)
        self.operations = deepcopy(operations)
        self.jobs = deepcopy(jobs)

    def get_machine_by_ID(self, ID):
        for m in self.machines:
            if m.ID == ID:
                return deepcopy(m)
        return None

    def get_operation_by_ID(self, ID):
        for op in self.operations:
            if op.ID == ID:
                return deepcopy(op)
        return None

    def get_job_by_ID(self, ID):
        for j in self.jobs:
            if j.ID == ID:
                return deepcopy(j)
        return None


class Solution:
    def __init__(self, operations: List[Operation], schedule=None):
        self.number_of_operations = len(operations)
        self.operations = sorted(operations, key=operator.attrgetter('ID'))
        if schedule is None:
            self.schedule = [op.job_number for op in self.operations]
        else:
            self.schedule = deepcopy(schedule)

        self.solution = {}
        for op in self.operations:
            self.solution[op.ID] = None

    def __eq__(self, other) -> bool:
        if not self.schedule == other.schedule or len(self.operations) != len(other.operations):
            return False

        for op_self, op_other in zip(self.operations, other.operations):
            if op_self.machine_ID != op_other.machine_ID:
                return False

        return True

    def update_solution_with_schedule(self):
        machines_queue = {}
        machines_remaining_occupancy_time = {}
        machines_currently_performed_operation = {}
        actual_operation_in_job = {}
        operations_left = set()
        for op in self.operations:
            operations_left.add(op.ID)
            machines_queue[op.machine_ID] = []
            machines_remaining_occupancy_time[op.machine_ID] = 0
            machines_currently_performed_operation[op.machine_ID] = None
            actual_operation_in_job[op.job_ID] = 1

        for job_number in self.schedule:
            job_ID = f"J{job_number}"
            operation_ID = f"O{job_number}_{actual_operation_in_job[job_ID]}"
            operation = self.find_operation_by_ID(operation_ID)
            machines_queue[operation.machine_ID].append(operation_ID)
            actual_operation_in_job[job_ID] += 1

        t = 0
        finished_operations = []
        while operations_left:
            for machine, queue in machines_queue.items():
                if machines_remaining_occupancy_time[machine] > 0:
                    machines_remaining_occupancy_time[machine] -= 1
                if machines_remaining_occupancy_time[machine] == 0 and machines_currently_performed_operation[machine] is not None:
                    finished_operations.append(machines_currently_performed_operation[machine])
                    operations_left.remove(machines_currently_performed_operation[machine])
                    machines_currently_performed_operation[machine] = None

            for machine, queue in machines_queue.items():
                if machines_remaining_occupancy_time[machine] == 0 and queue:
                    candidate_operation_ID = queue[0]
                    candidate_operation = self.find_operation_by_ID(candidate_operation_ID)
                    candidate_can_be_started = True
                    for op_number in range(1, candidate_operation.number):
                        if f"O{candidate_operation.job_number}_{op_number}" not in finished_operations:
                            candidate_can_be_started = False
                            break

                    if candidate_can_be_started:
                        self.solution[candidate_operation_ID] = t
                        queue.pop(0)
                        machines_currently_performed_operation[machine] = candidate_operation_ID
                        machines_remaining_occupancy_time[machine] = candidate_operation.duration

            t += 1

    def find_operation_by_ID(self, ID) -> Optional[Operation]:
        for op in self.operations:
            if op.ID == ID:
                return op
        return None

    def set_schedule(self, schedule) -> None:
        self.schedule = deepcopy(schedule)
        self.update_solution_with_schedule()

    def get_schedule(self) -> List[int]:
        return deepcopy(self.schedule)

    def get_solution(self):
        solution = {}
        for op in self.operations:
            solution[f"{op.ID} ({op.machine_ID})"] = self.solution[op.ID]
        return solution

    def save_solution(self, txt_path: str) -> None:
        solution = self.get_solution()
        jobs = 0
        solution_divided_to_jobs = {}
        for operation_machine, start in solution.items():
            op_m = operation_machine.split(" ")
            machine_nr = int(op_m[1][2: -1]) - 1  # starting with 0
            job_nr = int(op_m[0][1:].split("_")[0])
            operation_nr = int(op_m[0].split("_")[1])
            if job_nr not in solution_divided_to_jobs:
                jobs += 1
                solution_divided_to_jobs[job_nr] = []
            solution_divided_to_jobs[job_nr].append((operation_nr, machine_nr, start))

        text = ""
        for j in range(1, jobs + 1):
            line = ""
            operations = sorted(solution_divided_to_jobs[j], key=lambda x: x[0])
            for _, m, s in operations:
                line += f"{m} {s} "
            text += line[:-1] + "\n"

        with open(txt_path, "w") as file:
            file.write(text[:-1])

    def load_solution(self, txt_path: str) -> None:
        single_queue = []
        with open(txt_path, "r") as file:
            for job_nr, line in enumerate(file.readlines(), start=1):
                line = line.replace(" \n", "")
                line = line.replace("\n", "")
                line = line.replace("\t", " ")
                line_split = line.split(" ")
                op_nr = 1

                for i in range(0, len(line_split), 2):
                    m_ID = f"M{int(line_split[i]) + 1}"
                    start = int(line_split[i + 1])
                    op_ID = f"O{job_nr}_{op_nr}"
                    op = self.get_operation_by_ID(op_ID)
                    op.assign_to_machine(m_ID)
                    self.solution[op_ID] = start
                    op_nr += 1

                    single_queue.append((op_ID, start))

        single_queue.sort(key=lambda x: x[1])

        for ind, elem in enumerate(single_queue):
            op_ID = elem[0]
            job_nr = int(op_ID[1:].split("_")[0])
            self.schedule[ind] = job_nr

    def compute_solution_duration(self) -> int:
        self.update_solution_with_schedule()
        duration = 0
        for op_ID, op_start_time in self.solution.items():
            op = self.find_operation_by_ID(op_ID)
            duration = max(duration, op_start_time + op.duration)

        return duration

    def get_visualization(self):
        if self.solution is None:
            return

        df_dict = {}
        job_colors = {}
        colors = [
            'blueviolet',
            'cornflowerblue', 'crimson',
            'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkviolet', 'dimgray', 'dodgerblue',
            'firebrick',
            'gray', 'green',
            'hotpink',
            'indigo', 'indianred',
            'lightslategray',
            'maroon', 'mediumaquamarine', 'mediumvioletred', 'midnightblue',
            'navy',
            'olivedrab', 'orangered',
            'palevioletred', 'peru', 'purple',
            'rosybrown', 'royalblue',
            'saddlebrown', 'seagreen', 'slateblue', 'slategrey',
            'tan'
        ]
        for op_ID, start_time in self.solution.items():
            op = self.find_operation_by_ID(op_ID)
            machine_ID = op.machine_ID
            machine_nr = int(machine_ID[1:])

            if machine_nr not in df_dict:
                df_dict[machine_nr] = {
                    "machine": machine_ID,
                    "tasks": [(start_time, op.duration)],
                    "operations": [op_ID],
                    "jobs": [op.job_ID]
                }
            else:
                df_dict[machine_nr]["tasks"].append((start_time, op.duration))
                df_dict[machine_nr]["operations"].append(op_ID)
                df_dict[machine_nr]["jobs"].append(op.job_ID)

            if op.job_ID not in job_colors:
                job_colors[op.job_ID] = colors.pop(colors.index(rm.choice(colors)))

        df = pd.DataFrame(df_dict).T
        df.sort_index(inplace=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        index = 0
        machines = []
        for machine, row in df.iterrows():
            face_colors = [job_colors[job] for job in row["jobs"]]
            ax.broken_barh(xranges=row["tasks"], yrange=(index + 1, 0.5), facecolors=tuple(face_colors),
                           edgecolor='black')

            for task, label in zip(row["tasks"], row["operations"]):
                left_side, length = task
                ax.text(x=left_side + length / 2,
                        y=index + 1.25,
                        s=label,
                        ha='center',
                        va='center',
                        color='white',
                        )

            index += 1
            machines.append(row["machine"])

        ax.set_yticks([i + 1.25 for i in range(len(machines))])
        ax.set_yticklabels(machines)
        solution_duration = self.compute_solution_duration()

        if solution_duration < 25:
            step = 1
        elif 25 <= solution_duration < 50:
            step = 2
        elif 50 <= solution_duration < 100:
            step = 5
        elif 100 <= solution_duration < 200:
            step = 10
        elif 200 <= solution_duration < 300:
            step = 20
        elif 200 <= solution_duration < 400:
            step = 25
        elif 400 <= solution_duration < 500:
            step = 30
        elif 500 <= solution_duration < 600:
            step = 35
        elif 600 <= solution_duration < 700:
            step = 40
        elif 700 <= solution_duration < 800:
            step = 45
        elif 800 <= solution_duration < 1000:
            step = 50
        elif 1000 <= solution_duration < 1100:
            step = 60
        elif 1100 <= solution_duration < 1200:
            step = 70
        elif 1200 <= solution_duration < 1300:
            step = 80
        elif 1300 <= solution_duration < 1400:
            step = 90
        elif 1400 <= solution_duration < 2000:
            step = 100
        else:
            step = 200

        ax.set_xticks([i for i in range(0, solution_duration + step + 1, step)])

        return fig

    def visualize(self):
        if self.solution is None:
            return

        self.get_visualization()
        plt.show()

    def better_or_equal_than(self, other) -> bool:
        return self.compute_solution_duration() <= other.compute_solution_duration()

    def better_than(self, other) -> bool:
        return self.compute_solution_duration() < other.compute_solution_duration()

    def get_operation_by_ID(self, ID):
        for op in self.operations:
            if op.ID == ID:
                return op
        return None
