#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/../datasets"
cd "$SCRIPT_DIR/../datasets"

git clone --depth 1 --filter=blob:none --sparse https://github.com/AGI-Edgerunners/LLM-Adapters.git
cd LLM-Adapters
git sparse-checkout set dataset ft-training_set
cp -r dataset ../dataset
cp ft-training_set/commonsense_170k.json ../commonsense_170k.json
cd ..
rm -rf LLM-Adapters