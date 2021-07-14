import torch
from tsp import tours_length

def argsort_decode(Z, decoder, config, instance, tour):
    Z = torch.Tensor(Z).to(config.device)
    tour_idx = torch.argsort(Z, dim=1)
    costs = tours_length(instance.permute(0, 2, 1), tour_idx)
    return tour_idx, costs.tolist()
