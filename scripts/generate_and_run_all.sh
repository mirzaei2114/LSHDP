#!/bin/bash

# Define arrays of models and modes
models=("VIT_L_32" "VIT_L_16")
modes=("DDP" "DDP+FP" "FSDP" "FSDP+OFFLOAD" "LSHDP")

# Define common parameters
nnodes=4
nproc_per_node=1
node_rank=0
master_addr="172.20.253.19"
num_workers=4
num_iterations=8

# Loop through each combination of model and mode
for model in "${models[@]}"; do
  for mode in "${modes[@]}"; do
    # Create a script for this combination
    script_name="scripts/run_${model}_${mode}.sh"

    # Generate script content
    cat <<EOL > $script_name
#!/bin/bash

# Common parameters
nnodes=$nnodes
nproc_per_node=$nproc_per_node
node_rank=$node_rank
master_addr="$master_addr"
num_workers=$num_workers
num_iterations=$num_iterations

# Results directory
results_dir="results/${model}-${mode}"

echo "Running max_batch_sizes"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  max_batch_sizes.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --num_iterations \$num_iterations \
  --results_dir "\$results_dir"

# Use Python to load the tensor and calculate the global_total_batch_size
global_total_batch_size=\$(python3 - <<END
import torch

# Load the tensor from the .pt file
tensor = torch.load("\$results_dir/max_batch_sizes.pt")

# Calculate the sum of elements
print(tensor.min().item() * tensor.size(0))
END
)
echo "Global Total Batch Size: \$global_total_batch_size"

echo "Running best_batch_sizes_no_restriction"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  best_batch_sizes_no_restriction.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --num_iterations \$num_iterations \
  --results_dir "\$results_dir"

echo "Running best_batch_sizes_fixed_total"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  best_batch_sizes_fixed_total.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --num_iterations \$num_iterations \
  --results_dir "\$results_dir" \
  --total_batch_size \$global_total_batch_size

echo "Running trainer with fixed_total"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  trainer.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --results_dir "\$results_dir" \
  --batch_sizes_path fixed_total

echo "Running trainer with fixed_total and loss coefficient"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  trainer.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --results_dir "\$results_dir" \
  --batch_sizes_path fixed_total \
  --use_loss_coef

echo "Running trainer with no_restriction"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  trainer.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --results_dir "\$results_dir" \
  --batch_sizes_path no_restriction

echo "Running trainer with total batch size"
torchrun --nnodes \$nnodes \
  --nproc_per_node \$nproc_per_node \
  --node_rank \$node_rank \
  --master_addr \$master_addr \
  trainer.py \
  --model "$model" \
  --mode "$mode" \
  --num_workers \$num_workers \
  --results_dir "\$results_dir" \
  --total_batch_size \$global_total_batch_size
EOL

    # Make the script executable
    chmod +x $script_name

    # Run the generated script
    echo "Executing $script_name"
    ./$script_name
  done
done
