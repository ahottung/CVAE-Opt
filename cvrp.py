"""
This code is based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/tasks/cvrp.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import copy
import random

PRECISON = 0.00001


class CVRPDataset(Dataset):
    def __init__(self, size, problem_size, config, data, max_load=30, max_demand=9):
        super(CVRPDataset, self).__init__()

        self.config = config

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        self.size = size
        self.max_load = max_load
        self.max_demand = max_demand
        self.problem_size = problem_size
        self.instances = data[0]
        self.solutions = data[1]
        self.fixed_solution_length = int(self.problem_size * 1.5)

        assert len(self.instances) == len(self.solutions)
        assert len(self.instances) >= size


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        instance = np.array(self.instances[idx])
        solution = self.solutions[idx]
        assert len(solution) <= self.fixed_solution_length

        instance = torch.from_numpy(instance).to(self.config.device).float()

        # Create two symmetric solutions using augmentation based on one given solution

        solution_1 = solution_to_single_tour(solution)
        solution_tensor_1 = torch.zeros(self.fixed_solution_length)
        solution_tensor_1[:len(solution_1)] = torch.Tensor(solution_1)
        solution_tensor_1 = solution_tensor_1.long().to(self.config.device)

        solution_2 = solution_to_single_tour(solution)
        solution_tensor_2 = torch.zeros(self.fixed_solution_length)
        solution_tensor_2[:len(solution_2)] = torch.Tensor(solution_2)
        solution_tensor_2 = solution_tensor_2.long().to(self.config.device)

        return instance, solution_tensor_1, solution_tensor_2

def solution_to_single_tour(solution):
    """Transforms a list of tours to a single tour that returns to the depot (0) multiple times. Creates a randomly
    selected symmetric solution to the given solution."""

    solution = copy.deepcopy(solution)
    random.shuffle(solution)  # Shuffle order of the tours to get a random symmetric solution
    for i in range(len(solution)):
        if random.random() < 0.5:
            solution[i].reverse()  # Randomly either visit customer in the tour "forwards" or "backwards"
        solution[i].append(0)
    solution = [item for sublist in solution for item in sublist]
    solution.insert(0, 0)
    return solution


def update_mask(mask, dynamic, chosen_idx=None):
    """Updates the mask used to hide non-valid states.
    Parameters
    ----------
    dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
    """
    # Convert floating point to integers for calculations
    loads = dynamic[:, :, 0]  # (batch_size, seq_len)
    demands = dynamic[:, :, 1]  # (batch_size, seq_len)

    # If there is no positive demand left, we can end the tour.
    # Note that the first node is the depot, which always has a negative demand
    if demands.eq(0).all():
        return demands * 0.

    # Otherwise, we can choose to go anywhere where demand is > 0
    new_mask = demands.ne(0) * demands.lt(loads + PRECISON)

    # We should avoid traveling to the depot back-to-back
    mask_depot = chosen_idx.ne(0) | demands.sum(dim=1).eq(0)
    new_mask[:, 0] = mask_depot

    return new_mask.float()


def update_dynamic(instance, chosen_idx):
    """Updates the (load, demand) dataset values."""

    # Update the dynamic elements differently for if we visit depot vs. a city
    visit = chosen_idx.ne(0)
    depot = ~visit

    # Clone the dynamic variable so we don't mess up graph
    instance = instance.clone()
    all_loads = instance[:, :, 2]
    all_demands = instance[:, :, 3]


    demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1)).squeeze()

    # Across the minibatch - if we've chosen to visit a city, try to satisfy
    # as much demand as possible
    if visit.any():

        new_load = torch.clamp(all_loads[:, 0] - demand, min=0)

        visit_idx = visit.nonzero().squeeze()

        all_loads[:] = new_load.unsqueeze(1)
        all_demands[visit_idx, chosen_idx[visit_idx]] = 0
        all_demands[visit_idx, 0] = -1. + new_load[visit_idx]

    # Return to depot to fill vehicle load
    if depot.any():
        depot_idx = depot.nonzero().squeeze()
        all_loads[depot_idx] = 1.
        all_demands[depot_idx, 0] = 0.

    return instance




def tours_length(locations, tours):
    y = torch.gather(locations[:, :, :2], 1, tours.unsqueeze(2).expand(-1, -1, 2))

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()