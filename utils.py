import numpy as np
import zipfile
import pickle

def read_instance_data_tsp(problem_size, nb_instances, instance_file, solution_file, offset=0):
    instances = []
    solutions = []
    with zipfile.ZipFile(instance_file) as instance_zip:
        with zipfile.ZipFile(solution_file) as solution_zip:
            instances_list = instance_zip.namelist()
            solutions_list = solution_zip.namelist()
            assert len(instances_list) == len(solutions_list)
            instances_list.sort()
            solutions_list.sort()
            i = offset
            while len(instances) < nb_instances:
                if instances_list[i].endswith('/'):
                    i += 1
                    continue

                #Read instance data
                f = instance_zip.open(instances_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                instance = np.zeros((problem_size, 2))
                ii = 0
                while not lines[ii].startswith("NODE_COORD_SECTION"):
                    ii += 1
                ii += 1
                header_lines = ii
                while ii < len(lines):
                    line = lines[ii]
                    if line == 'EOF':
                        break
                    line = line.replace('\t', " ").split(" ")
                    x = line[1]
                    y = line[2]
                    instance[ii-header_lines] = [x, y]
                    ii += 1

                instance = np.array(instance) / 1000000
                instances.append(instance)

                #Read solution data
                f = solution_zip.open(solutions_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                tour = [int(l) for ll in lines[1:] for l in ll.split(' ')]

                solutions.append(tour)
                i += 1

    return instances, solutions


def read_instance_data_cvrp(problem_size, nb_instances, instance_file, solution_file, offset=0):
    instances = []
    solutions = []
    with zipfile.ZipFile(instance_file) as instance_zip:
        with zipfile.ZipFile(solution_file) as solution_zip:
            instances_list = instance_zip.namelist()
            solutions_list = solution_zip.namelist()
            assert len(instances_list) == len(solutions_list)
            instances_list.sort()
            solutions_list.sort()
            i = offset
            while len(instances) < nb_instances:
                if instances_list[i].endswith('/'):
                    i += 1
                    continue

                #Read instance data
                f = instance_zip.open(instances_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                ii = 0

                while ii < len(lines):
                    line = lines[ii]
                    if line.startswith("DIMENSION"):
                        dimension = int(line.split(':')[1])
                    elif line.startswith("CAPACITY"):
                        capacity = int(line.split(':')[1])
                    elif line.startswith('NODE_COORD_SECTION'):
                        locations = np.loadtxt(lines[ii + 1:ii + 1 + dimension], dtype=float)
                        ii += dimension
                    elif line.startswith('DEMAND_SECTION'):
                        demand = np.loadtxt(lines[ii + 1:ii + 1 + dimension], dtype=float)
                        ii += dimension

                    ii += 1

                locations = locations[:, 1:] / 1000000
                demand = demand[:, 1:] / capacity
                loads = np.ones((len(locations), 1))
                instance = np.concatenate((locations, loads, demand), axis=1)
                instances.append(instance)

                #Read solution data
                f = solution_zip.open(solutions_list[i], "r")
                solution = []
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                ii = 0
                while ii < len(lines):
                    line = lines[ii]
                    ii += 1
                    if not line.startswith("Route"):
                        continue
                    line = line.split(':')[1]
                    tour = [int(l) for l in line[1:].split(' ')]
                    solution.append(tour)


                solutions.append(solution)
                i += 1

    return instances, solutions

def read_instance_data(config):
    offset = max(config.network_validation_size, config.search_validation_size)

    if config.problem == "TSP":
        training_data = read_instance_data_tsp(config.problem_size, config.epoch_size, config.instances_path,
                                           config.solutions_path, offset)
        validation_data = read_instance_data_tsp(config.problem_size, offset, config.instances_path,
                                             config.solutions_path)
    elif config.problem == "CVRP":
        training_data = read_instance_data_cvrp(config.problem_size, config.epoch_size, config.instances_path,
                                               config.solutions_path, offset)
        validation_data = read_instance_data_cvrp(config.problem_size, offset, config.instances_path,
                                                 config.solutions_path)
    return training_data, validation_data



def read_instance_pkl(config):
    with open(config.instances_path, 'rb') as f:
        instances_data = pickle.load(f)

    if config.problem == "TSP":
        return instances_data
    elif config.problem == "CVRP":
        instances = []
        for instance in instances_data:
            instance_np = np.zeros((config.problem_size + 1, 4))
            instance_np[0, :2] = instance[0]  # depot location
            instance_np[1:, :2] = instance[1]  # customer locations
            instance_np[:, 2] = 1  # loads
            instance_np[1:, 3] = np.array(instance[2]) / instance[3]  # customer demands
            instance_np[0, 3] = 0  # depot demand
            instances.append(instance_np)
        return instances
