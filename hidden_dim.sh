#!/bin/bash

# Array of hidden_dim values
hidden_dim_values=(4, 16, 256, 1024)

# Loop through each value and execute the command
for hidden_dim in "${hidden_dim_values[@]}"
do
  echo "Running with hidden_dim=${hidden_dim}"
  python run_graph_classification.py --layer_type GCN --hidden_dim "$hidden_dim" --dataset mutag
done