#!/bin/bash

echo "Starting all Experiment"

# execute first script
echo "Starting script 1: run_cd_benchmark_rn50.sh"
bash ./scripts/run_cd_benchmark_rn50.sh
if [ $? -ne 0 ]; then
    echo "Error: run_cd_benchmark_rn50.sh failed. Exiting."
    exit 1
fi
echo "Completed script 1: run_cd_benchmark_rn50.sh"

# execute second script
echo "Starting script 2: run_cd_benchmark_vit.sh"
bash ./scripts/run_cd_benchmark_vit.sh
if [ $? -ne 0 ]; then
    echo "Error: run_cd_benchmark_vit.sh failed. Exiting."
    exit 1
fi
echo "Completed script 2: run_cd_benchmark_vit.sh"

# execute third script
echo "Starting script 3: run_ood_benchmark_rn50.sh"
bash ./scripts/run_ood_benchmark_rn50.sh
if [ $? -ne 0 ]; then
    echo "Error: run_ood_benchmark_rn50.sh failed. Exiting."
    exit 1
fi
echo "Completed script 3: run_ood_benchmark_rn50.sh"

# execute fourth script
echo "Starting script 4: run_ood_benchmark_vit.sh"
bash ./scripts/run_ood_benchmark_vit.sh
if [ $? -ne 0 ]; then
    echo "Error: run_ood_benchmark_vit.sh failed. Exiting."
    exit 1
fi
echo "Completed script 4: run_ood_benchmark_vit.sh"

echo "Experiment completed successfully."
