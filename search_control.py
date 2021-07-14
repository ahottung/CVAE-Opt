import torch
import tsp, cvrp
import numpy as np
import time
from de import minimize
import logging
import os


def decode(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx)
    return tour_idx, costs.tolist()


def evaluate(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx)
    return costs.tolist()


def solve_instance_de(model, instance, config, cost_fn):
    batch_size = config.search_batch_size
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size, config)

    result_cost, result_tour = minimize(decode, (model, config, instance, cost_fn), config.search_space_bound,
                                        config.search_space_size, popsize=batch_size,
                                        mutate=config.de_mutate, recombination=config.de_recombine,
                                        maxiter=config.search_iterations, maxtime=config.search_timelimit)
    solution = decode(np.array([result_tour] * batch_size), model, config, instance, cost_fn)[0][0].tolist()
    return result_cost, solution


def solve_instance_set(model, config, instances, solutions=None, verbose=True):
    model.eval()

    if config.problem == "TSP":
        cost_fn = tsp.tours_length
    elif config.problem == "CVRP":
        cost_fn = cvrp.tours_length
        if solutions:
            solutions = [cvrp.solution_to_single_tour(solution) for solution in solutions]

    gap_values = np.zeros((len(instances)))
    cost_values = []
    runtime_values = []
    for i, instance in enumerate(instances):
        start_time = time.time()
        objective_value, solution = solve_instance_de(model, instance, config, cost_fn)
        runtime = time.time() - start_time

        if solutions:
            optimal_value = cost_fn(torch.Tensor(instance).unsqueeze(0),
                                    torch.Tensor(solutions[i]).long().unsqueeze(0)).item()
            print(objective_value, optimal_value)
            print("Opt " + str(solutions[i]))
            gap = (objective_value / optimal_value - 1) * 100
            print("Gap " + str(gap) + "%")
            gap_values[i] = gap
        cost_values.append(objective_value)
        print("Costs " + str(objective_value))
        runtime_values.append(runtime)

    if not solutions and verbose:
        results = np.array(list(zip(cost_values, runtime_values)))
        np.savetxt(os.path.join(config.output_path, "search", 'results.txt'), results, delimiter=',', fmt=['%s', '%s'],
                   header="cost, runtime")
        logging.info("Final search results:")
        logging.info(f"Mean costs: {np.mean(cost_values)}")
        logging.info("Mean std: {}".format(np.mean(np.std(gap_values))))
        logging.info(f"Mean runtime: {np.mean(runtime_values)}")

    return np.mean(gap_values), np.mean(runtime_values), cost_values
