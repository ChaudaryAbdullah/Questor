#!/bin/bash

# Simple script to run all component permutations for the ablation study

echo "=========================================================="
echo "Starting Ablation Study"
echo "=========================================================="

echo -e "\n\n--> 1. Running ALL Components (Baseline)"
python3 run_unified.py

echo -e "\n\n--> 2. Running without Agents"
python3 run_unified.py --no-agents

echo -e "\n\n--> 3. Running without Structured Pipeline"
python3 run_unified.py --no-structured

echo -e "\n\n--> 4. Running without Unstructured Pipeline"
python3 run_unified.py --no-unstructured

echo -e "\n\n--> 5. Running WITHOUT Structured AND WITHOUT Unstructured (Agents Only)"
python3 run_unified.py --no-structured --no-unstructured

echo -e "\n\n--> 6. Running WITHOUT Unstructured AND WITHOUT Agents (Structured Only)"
python3 run_unified.py --no-unstructured --no-agents

echo -e "\n\n--> 7. Running WITHOUT Structured AND WITHOUT Agents (Unstructured Only)"
python3 run_unified.py --no-structured --no-agents

echo -e "\n\n=========================================================="
echo "Ablation Study Completed!"
echo "=========================================================="
