#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../reference-code/commonsense_reasoning"

git clone --depth 1 --filter=blob:none --sparse https://github.com/AGI-Edgerunners/LLM-Adapters.git
cd LLM-Adapters
git sparse-checkout set dataset ft-training_set
cp -r dataset ../dataset
cp ft-training_set/commonsense_170k.json ../commonsense_170k.json