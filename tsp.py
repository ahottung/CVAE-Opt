"""
This code is based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/tasks/cvrp.py
"""


import numpy as np
from torch.utils.data import Dataset
import torch


class TSPDataset(Dataset):

    def __init__(self, size, problem_size, config, data):
        self.size = size
        self.problem_size = problem_size
        self.instances = data[0]
        self.solutions = data[1]
        self.config = config

        assert len(self.instances) == len(self.solutions)
        assert len(self.instances) >= size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        locations = self.instances[idx]
        tour = np.array(self.solutions[idx])

        locations = torch.from_numpy(locations).to(self.config.device).float()
        tours = torch.from_numpy(tour).long().to(self.config.device)

        # Create two symmetric solutions using augmentation based on one given solution
        shuffle_index_1 = np.random.randint(0, self.config.problem_size)
        solution_1 = torch.cat((tours[shuffle_index_1:], tours[:shuffle_index_1]))
        shuffle_index_2 = np.random.randint(0, self.config.problem_size)
        solution_2 = torch.cat((tours[shuffle_index_2:], tours[:shuffle_index_2]))

        return locations, solution_1, solution_2



def tours_length(locations, tours):
    locations_tour_input = torch.gather(locations, 1, tours.unsqueeze(2).expand_as(locations))
    y = torch.cat((locations_tour_input, locations_tour_input[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

def update_dynamic(instance, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    instance = instance.clone()
    instance[:, :, 2].scatter_(1, chosen_idx.unsqueeze(1), 0)
    return instance
