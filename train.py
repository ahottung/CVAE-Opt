from config_train import get_config
from torch.utils.data import DataLoader
from tsp import TSPDataset
from cvrp import CVRPDataset
import torch
import numpy as np
import time
import datetime
import os
import logging
import sys
from utils import read_instance_data
from search_control import solve_instance_set
from VAE_8 import VAE_8


def calculate_RC_loss(tour_logp):
    RC = - tour_logp.sum()
    return RC


def calculate_KLD_loss(mean, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD


def evaluate_network(config, model, validation_dataloader, epoch_idx):
    model.eval()
    loss_RC_values = []
    loss_KLD_values = []
    abs_Z_values = []
    for batch_id, batch in enumerate(validation_dataloader):
        instances, solutions_1, solutions_2 = batch

        with torch.no_grad():
            output, mean, log_var, Z, tour_idx, tour_logp = VAEModel(instances, solutions_1, solutions_2, config)
        loss_RC = calculate_RC_loss(tour_logp)
        loss_KLD = calculate_KLD_loss(mean, log_var)

        loss_RC_values.append(loss_RC.item())
        loss_KLD_values.append(loss_KLD.item())
        abs_Z = torch.abs(Z)  # Absolute coordinates of points in latent space (Z)
        abs_Z_values.append(abs_Z.cpu().numpy())

    abs_Z_values = np.array(abs_Z_values).flatten()

    # The bounds of the search space are defined as a percentile of the absolute latent variable coordinates
    new_bound = np.percentile(abs_Z_values, config.q_percentile).item()

    logging.info(f'[Network Validation] Loss_RC: {np.mean(loss_RC_values)} Loss_KLD: {np.mean(loss_KLD_values)}')

    model.train()

    return new_bound


def train_epoch(model, config, epoch_idx, training_dataloader, optimizer):
    model.train()

    start_time = time.time()

    loss_RC_values = []
    loss_KLD_values = []

    for batch_id, batch in enumerate(training_dataloader):
        print("Batch {}/{}".format(batch_id, int(config.epoch_size / config.batch_size)))

        # Get an instance and two symmetric solutions (see the paragraph symmetry breaking in the paper)
        instances, solutions_1, solutions_2 = batch

        # Forward pass
        output, mean, log_var, Z, tour_idx, tour_logp = VAEModel(instances, solutions_1, solutions_2, config)

        # Calculate weighted loss
        loss_RC = calculate_RC_loss(tour_logp)
        loss_KLD = calculate_KLD_loss(mean, log_var)
        loss = loss_RC + loss_KLD * config.KLD_weight

        # Update network weights
        optimizer.zero_grad()
        assert not torch.isnan(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        loss_RC_values.append(loss_RC.item())
        loss_KLD_values.append(loss_KLD.item())

    epoch_duration = time.time() - start_time
    logging.info(
        "Finished epoch {}, took {} s".format(epoch_idx, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    logging.info("Epoch Loss_RC {}, Epoch Loss_KLD {}".format(np.mean(loss_RC_values), np.mean(loss_KLD_values)))


def train(model, config):
    assert config.nb_epochs > 20
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    training_data, validation_data = read_instance_data(config)

    if config.problem == "TSP":
        training_dataset = TSPDataset(config.epoch_size, config.problem_size, config, training_data)
        validation_dataset = TSPDataset(config.network_validation_size, config.problem_size, config, validation_data)
    elif config.problem == "CVRP":
        training_dataset = CVRPDataset(config.epoch_size, config.problem_size, config, training_data)
        validation_dataset = CVRPDataset(config.network_validation_size, config.problem_size, config, validation_data)

    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=0, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=0, shuffle=True)

    best_avg_gap = np.inf
    for epoch_idx in range(1, config.nb_epochs + 1):
        train_epoch(model, config, epoch_idx, training_dataloader, optimizer)

        # Validate and save model every 20 epochs
        if epoch_idx % 20 == 0:
            logging.info("Start validation")
            # Evaluate the network performance on the validation set
            new_bound = evaluate_network(config, VAEModel, validation_dataloader, epoch_idx)

            config.search_space_bound = new_bound
            logging.info(f"Setting search space bounds to {new_bound:2.3f}")

            # Evaluate the search performance of the current model on a small subset of the validation set
            avg_gap, avg_runtime, _ = solve_instance_set(VAEModel, config,
                                                         validation_data[0][: config.search_validation_size]
                                                         , validation_data[1][:config.search_validation_size])

            # If the average gap is improved, save the model
            if avg_gap < best_avg_gap:
                best_avg_gap = avg_gap
                model_data = {
                    'parameters': model.state_dict(),
                    'code_version': VERSION,
                    'problem': config.problem,
                    'problem_size': config.problem_size,
                    'Z_bound': new_bound,
                    'avg_gap': avg_gap,
                    'training_epochs': epoch_idx,
                    'model': "VAE_final"
                }

                torch.save(model_data, os.path.join(config.output_path, "models",
                                                    "model_{0}.pt".format(run_id, epoch_idx)))

            logging.info("Validation gap: {}% ({}%), Avg. Runtime: {}".format(avg_gap, best_avg_gap, avg_runtime))

    # Save the last model after the end of the training
    torch.save(model_data, os.path.join(config.output_path, "models",
                                        "model_{0}_final.pt".format(run_id, epoch_idx)))
    logging.info("Training finished")


VERSION = "0.4.0"
if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    now = datetime.datetime.now()

    config = get_config()

    if config.output_path == "":
        config.output_path = os.getcwd()
    config.output_path = os.path.join(config.output_path, "runs", "run_" + str(now.day) + "." + str(now.month) +
                                      "." + str(now.year) + "_" + str(run_id))
    os.makedirs(os.path.join(config.output_path, "models"))

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Started Training Run")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("Version: {0}".format(VERSION))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    VAEModel = VAE_8(config).to(config.device)

    train(VAEModel, config)
