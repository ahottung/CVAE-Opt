from config_search import get_config

import torch
import numpy as np
import datetime
import os
import logging
import sys
import train
import search_control
from utils import read_instance_pkl
from VAE_8 import VAE_8

if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    now = datetime.datetime.now()

    config = get_config()

    if config.output_path == "":
        config.output_path = os.getcwd()
    config.output_path = os.path.join(config.output_path, "runs", "run_" + str(now.day) + "." + str(now.month) +
                                      "." + str(now.year) + "_" + str(run_id))

    os.makedirs(os.path.join(config.output_path, "search"))

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')

    logging.info("Started Search Run")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("Version: {0}".format(train.VERSION))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    model_data = torch.load(config.model_path, config.device)

    config.search_space_bound = model_data['Z_bound']
    logging.info(f"Setting search space bound to {config.search_space_bound}")

    if not config.problem:
        config.problem = model_data['problem']
    if not config.problem_size:
        config.problem_size = model_data['problem_size']

    model = VAE_8(config).to(config.device)
    model.load_state_dict(model_data['parameters'])
    model.eval()

    instances = read_instance_pkl(config)

    _, avg_runtime, costs = search_control.solve_instance_set(model, config, instances)
