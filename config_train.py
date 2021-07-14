import argparse
import torch



def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="CVAE-Opt Training")

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--problem', type=str, default='TSP')
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--search_batch_size', default=600, type=int)
    parser.add_argument('--epoch_size', type=int, default=93440, help='Number of instances used for training')
    parser.add_argument('--nb_epochs', default=300, type=int)
    parser.add_argument('--search_validation_size', default=100, type=int)
    parser.add_argument('--network_validation_size', default=6400, type=int)
    parser.add_argument('--search_space_size', default=100, type=int)
    parser.add_argument('--KLD_weight', default=None, type=float)  # Beta in the paper
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--instances_path', type=str, default="data/tsp/training/tsp_100_instances.zip")
    parser.add_argument('--solutions_path', type=str, default="data/tsp/training/tsp_100_solutions.zip")
    parser.add_argument('--q_percentile', default=99, type=float)
    parser.add_argument('--search_timelimit', default=600, type=int)
    parser.add_argument('--search_iterations', default=300, type=int)

    # Differential Evolution
    parser.add_argument('--de_mutate', default=0.3, type=float)
    parser.add_argument('--de_recombine', default=0.95, type=float)

    config = parser.parse_args()

    config.device = torch.device(config.device)

    return config
