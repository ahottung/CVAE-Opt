import argparse
import torch



def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="CAVE-Opt Search")

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model_path', type=str, default='', required=True)
    parser.add_argument('--problem', type=str, default=None)
    parser.add_argument("--problem_size", type=int, default=None)
    parser.add_argument('--search_batch_size', default=600, type=int)
    parser.add_argument('--instances_path', type=str, default="")
    parser.add_argument('--search_timelimit', default=600, type=int)
    parser.add_argument('--search_space_size', default=100, type=int)  # Nb. dimensions of search space
    parser.add_argument('--search_iterations', default=300, type=int)

    # Differential Evolution
    parser.add_argument('--de_mutate', default=0.3, type=float)
    parser.add_argument('--de_recombine', default=0.95, type=float)

    config = parser.parse_args()
    config.device = torch.device(config.device)

    return config
