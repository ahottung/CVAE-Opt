# CVAE-Opt

This repository contains the code used for the experiments in the paper "Learning a Latent Search Space for Routing Problems using Variational Autoencoders" (https://openreview.net/pdf?id=90JprVrJBO).

CVAE-Opt learns a continuous latent search space for routing problems that can be searched by any continuous optimization method. It is based on a conditional variational autoencoder (CVAE) that learns to map solutions to routing problem instances to a continuous, n-dimensional space.

### Instances 

The instances used for the experiments in the paper can be found in the folder "instances".  The test instances have been generated with the generator from https://github.com/wouterkool/attention-learn-to-route. We also provide the large training instance sets and the generated, corresponding high-quality solutions using  Git Large File Storage (LFS). If you do not have Git LFS installed, you can also download the files manually. 

### Paper
```
@inproceedings{hottung2020learning,
  title={Learning a Latent Search Space for Routing Problems using Variational Autoencoders},
  author={Hottung, Andr{\'e} and Bhandari, Bhanu and Tierney, Kevin},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## Requirements

CVAE-Opt requires python (>= 3.6) and the following python packages:

- numpy
- pytorch (we used version 1.3.1 for evaluating CVAE-Opt)

## Quick Start

By default, CVAE-Opt uses the GPU. If you want to run CVAE-Opt on the CPU only, use the `--device cpu` option.

### Solving the test instance

To solve the the traveling salesperson problem (TSP) test instances instance with 20 nodes using the provided pre-trained models run the following command: 


```bash
python3 search.py --problem TSP --problem_size 20 --instances_path instances/tsp/test/tsp20_test_small_seed1235.pkl --model_path models/tsp_20_model_74446.pt
```

The duration of the search can be limited either by iterations (e.g., ``--search_iterations 100``) or by runtime  in seconds (e.g., ``--search_timelimit 180``).

### Training a new model

Training a new model requires a training set of representative instances and corresponding high-quality solutions. We provide all training data sets that we used for the training of our models. To train a new model for TSP instances with 20 nodes using a KLD_weight (beta in the paper) of 0.0001 run the following command:

```bash
python3 train.py --instances_path instances/tsp/training/tsp_20_instances.zip
--solutions_path instances/tsp/training/tsp_20_solutions.zip --problem TSP --problem_size 20 --KLD_weight 0.0001
```



## Acknowledgements

The code is originally based on https://github.com/mveres01/pytorch-drl4vrp which is a great starting point for learning/implementing deep reinforcement learning approaches for vehicle routing problems.  The implementation of the differential evolution algorithm is based on https://github.com/nathanrooy/differential-evolution-optimization.