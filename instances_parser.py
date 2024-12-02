from process_model import Operation, Machine, Job, Process
from copy import deepcopy


class ClassicJobShopParser:
    def __init__(self):
        pass

    # def parse(self, txt_path: str):
    #     return self.parse_txt_with_standard_specification(txt_path)

    @staticmethod
    def parse(txt_path: str):
        operations = []
        jobs = []
        machines = []
        job_cnt = 1
        with open(txt_path, "r") as file:
            first_line = True
            for line in file.readlines():
                line = line.replace(" \n", "")
                line = line.replace("\n", "")
                line = line.replace("\t", " ")
                line_split = line.split(" ")
                if first_line:
                    nr_of_jobs = int(line_split[0])
                    nr_of_machines = int(line_split[1])
                    for j in range(nr_of_jobs):
                        jobs.append(Job(ID=f"J{j+1}", operations=[]))

                    for m in range(nr_of_machines):
                        machines.append(Machine(ID=f"M{m+1}"))
                    first_line = False
                    continue

                operations_in_job = []
                op_nr = 1
                for i in range(0, len(line_split), 2):
                    mach = machines[int(line_split[i])]
                    op = Operation(ID=f"O{job_cnt}_{op_nr}", duration_tab={mach.ID: int(line_split[i+1])}, machine_ID=mach.ID)
                    operations_in_job.append(op)
                    op_nr += 1

                jobs[job_cnt-1].operations = deepcopy(operations_in_job)
                operations.extend(operations_in_job)
                job_cnt += 1

        return Process(machines, operations, jobs)


class FlexibleJobShopParser:
    def __init__(self):
        pass

    # def parse(self, txt_path: str):
    #     return self.parse_txt_with_standard_specification(txt_path)

    @staticmethod
    def parse(txt_path: str):
        operations = []
        jobs = []
        machines = []
        job_cnt = 1
        with open(txt_path, "r") as file:
            first_line = True
            for line in file.readlines():
                if line[0] == "\n":
                    continue

                if line[0] == ' ':
                    line = line[1:]
                while "  " in line:
                    line = line.replace("  ", " ")
                line = line.replace(" \n", "")
                line = line.replace("\n", "")
                line = line.replace("\t", " ")
                line_split = line.split(" ")
                if first_line:
                    nr_of_jobs = int(line_split[0])
                    nr_of_machines = int(line_split[1])
                    for j in range(nr_of_jobs):
                        jobs.append(Job(ID=f"J{j + 1}", operations=[]))

                    for m in range(nr_of_machines):
                        machines.append(Machine(ID=f"M{m + 1}"))
                    first_line = False
                    continue

                operations_in_job = []
                nr_of_operations = int(line_split.pop(0))
                for op_nr in range(1, nr_of_operations + 1):
                    duration_tab = {}
                    nr_of_machines_that_can_process_op = int(line_split.pop(0))
                    for _ in range(nr_of_machines_that_can_process_op):
                        mach = machines[int(line_split.pop(0)) - 1]
                        duration = int(line_split.pop(0))
                        duration_tab[mach.ID] = duration

                    op = Operation(ID=f"O{job_cnt}_{op_nr}", duration_tab=deepcopy(duration_tab), machine_ID=None)
                    operations_in_job.append(op)

                jobs[job_cnt - 1].operations = deepcopy(operations_in_job)
                operations.extend(operations_in_job)
                job_cnt += 1

        return Process(machines, operations, jobs)


class JobShopParser:
    def __init__(self):
        self.classic_parser = ClassicJobShopParser()
        self.flexible_parser = FlexibleJobShopParser()

    def parse_classic_job_shop_instance(self, txt_path: str):
        return self.classic_parser.parse(txt_path)

    def parse_flexible_job_shop_instance(self, txt_path: str):
        return self.flexible_parser.parse(txt_path)
