# LSHDP: Locally Sharded Heterogeneous Data Parallel for Distributed Deep Learning

[![Paper](https://img.shields.io/badge/Published%20in-Parallel%20Computing%20Journal-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0167819125000407)

This repository contains the implementation of **LSHDP (Locally Sharded Heterogeneous Data Parallel)**, a novel approach for distributed deep learning on heterogeneous infrastructures. The method is designed to optimize GPU utilization and minimize node waiting times in environments with limited bandwidth and varying computational power across nodes.

LSHDP builds on the principles of data parallelism and consists of two independently applicable components:

- **HDP (Heterogeneous Data Parallel)**: Designed for heterogeneous infrastructures, HDP assigns batch sizes to nodes at the start of training based on their computational capacities. This allocation ensures minimum training time while accommodating diverse hardware capabilities.
- **LSDP (Locally Sharded Data Parallel)**: Focused on minimizing bandwidth usage, LSDP provides a communication-efficient alternative to established methods like DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel).

## Performance Improvements
The LSHDP method demonstrates significant performance improvements in heterogeneous environments with low inter-node communication speeds:
- **35.39% speed improvement** over Data Parallel methods.
- **52.57% speed improvement** over Fully Sharded Data Parallel (FSDP) methods.

## Repository Structure
```
.
├── best_batch_sizes_fixed_total.py   # Script for optimizing batch sizes with a fixed total
├── best_batch_sizes_no_restriction.py # Script for optimizing batch sizes without restrictions
├── heterogeneous_distributed_sampler.py # Custom sampler for heterogeneous distributed training
├── logger.py                         # Logging utilities
├── max_batch_sizes.py                # Script to find maximum batch sizes per rank
├── profiler.py                       # Profiler for measuring execution time and memory usage
├── trainer.py                        # Training script for distributed deep learning
├── utils.py                          # Utility functions for distributed setup and model handling
└── scripts/
    └── generate_and_run_all.sh       # Script to generate and execute all experiments
```

## Getting Started

### Prerequisites
- PyTorch 2.5.1 (tested version, but other compatible versions should work)
- TorchVision 0.20 (tested version, but other compatible versions should work)
- NVIDIA GPUs with CUDA support
- Multi-node setup with heterogeneous GPUs (optional)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/LSHDP.git
   cd LSHDP
   ```

2. Install dependencies:
   ```bash
   pip install tqdm
   pip install torch==2.5.1 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
   ```

### Usage

1. **Automated Experiment Execution**:
   To run all experiments automatically, use the provided script:
   ```bash
   bash scripts/generate_and_run_all.sh
   ```
   This script generates and executes all necessary scripts for the experiments, including finding maximum batch sizes, optimizing batch sizes, and training the model.

2. **Manual Execution**:
   Alternatively, you can perform the steps manually:

   - **Find Maximum Batch Sizes**:
     Run the [`max_batch_sizes.py`](max_batch_sizes.py) script to determine the maximum batch sizes for each rank:
     ```bash
     python max_batch_sizes.py --model VIT_L_32 --mode FSDP --num_workers 4 --num_iterations 8 --results_dir results/
     ```

   - **Optimize Batch Sizes**:
     - Without restriction:
       ```bash
       python best_batch_sizes_no_restriction.py --model VIT_L_32 --mode FSDP --num_workers 4 --num_iterations 8 --results_dir results/
       ```
     - With fixed total:
       ```bash
       python best_batch_sizes_fixed_total.py --model VIT_L_32 --mode FSDP --num_workers 4 --num_iterations 8 --total_batch_size 128 --results_dir results/
       ```

   - **Train the Model**:
     Train the model using the optimized batch sizes:
     ```bash
     python trainer.py --model VIT_L_32 --mode FSDP --batch_sizes_path fixed_total --num_workers 4 --results_dir results/
     ```

## Citation

The work is published in the **Parallel Computing Journal** and can be cited as follows:

```
@article{MIRZAEI2025103164,
title = {LSHDP: Locally Sharded Heterogeneous Data Parallel for Distributed Deep Learning},
journal = {Parallel Computing},
pages = {103164},
year = {2025},
issn = {0167-8191},
doi = {https://doi.org/10.1016/j.parco.2025.103164},
url = {https://www.sciencedirect.com/science/article/pii/S0167819125000407},
author = {Motahhare Mirzaei and Mehrdad Ashtiani and Mohammad Javad Pirhadi and Sauleh Eetemadi},
keywords = {Distributed training, neural networks, heterogeneous infrastructure, FSDP},
abstract = {In today’s world, pre-trained models such as GPT-3 and Llama 3.1, along with the use of transformers, recognized as large AI models, have gained significant importance. To accelerate the training of these models, distributed training has become a fundamental approach. This method enables the execution of model training across multiple GPUs, which is particularly essential for models that require more data and training time. Despite past advancements, achieving optimal utilization of GPU capacity remains a major challenge, especially in academic environments that often feature heterogeneous infrastructures and limited bandwidth between nodes, which do not align with the assumptions of existing methods. In previous methods, the node with the lowest computational power is considered the bottleneck, leading to computational slowdowns and increased waiting times for other nodes. This study addresses the issue by adjusting batch sizes to minimize node waiting times. This approach improves the efficiency of node utilization without reducing the convergence speed. Moreover, to address GPU memory limitations, existing methods often rely on high-speed inter-node communication. This reliance increases training time in scenarios with low network bandwidth (e.g., 1 Gb/s). This research mitigates this challenge using the LSDP (Locally Sharded Data Parallel) method, which leverages CPU memory instead of inter-node communication. Finally, by combining these two strategies, the LSHDP (Locally Sharded Heterogeneous Data Parallel) solution is introduced which is suitable for heterogeneous infrastructures with low inter-node communication speeds. Experiments demonstrate that this method outperforms previous approaches in such environments, achieving improvements of 35.39% and 52.57% in terms of speed compared to data-parallel and Fully Sharded Data Parallel (FSDP) methods respectively.}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
This work was supported by the authors' institutions and the Parallel Computing Journal. For more details, refer to the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167819125000407).
