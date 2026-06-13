#!/bin/bash

set -e

echo "Starting..."

if [ ! -d ".venv" ]; then
    echo -e "\n[1/6] Generating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo -e "\n[2/6] Instaling requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n[3/6] Downloading Kaggle dataset and creating mapping.json..."
python includes.py

echo -e "\n[4/6] Training ResNet18..."
python resnet18.py

echo -e "\n[5/6] Exporting to ONNX..."
python export_to_onnx.py

echo -e "\n[6/6] Running benchmarks..."
python benchmark.py


