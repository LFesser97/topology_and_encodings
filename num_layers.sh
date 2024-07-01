#!/bin/bash

# Array of num_layers values
num_layers_values=(2 4 8 16 32)

# Loop through each value and execute the command
for num_layers in "${num_layers_values[@]}"
do
  echo "Running with num_layers=${num_layers}"
  python run_graph_classification.py --layer_type GCN --num_layers "$num_layers" --dataset enzymes
done