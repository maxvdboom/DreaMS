#!/bin/bash
# WandB Setup Script for GPU Cluster

echo "Setting up WandB for DreaMS training..."

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "WandB not found. Installing..."
    pip install wandb
fi

# Check if already logged in
if wandb whoami &> /dev/null; then
    echo "Already logged in to WandB as: $(wandb whoami)"
else
    echo "Please log in to WandB. You have several options:"
    echo ""
    echo "Option 1: Login interactively (recommended for first-time setup)"
    echo "  Run: wandb login"
    echo ""
    echo "Option 2: Use API key directly"
    echo "  Run: wandb login YOUR_API_KEY"
    echo ""
    echo "Option 3: Set environment variable (for automated scripts)"
    echo "  export WANDB_API_KEY=your_api_key_here"
    echo ""
    echo "Option 4: Use offline mode (no internet required)"
    echo "  export WANDB_MODE=offline"
    echo ""
    echo "You can get your API key from: https://wandb.ai/authorize"
fi

# Check WandB entity configuration
echo ""
echo "Current WandB configuration:"
echo "Entity: mass-spec-ml (default for DreaMS)"
echo "You can change this by adding --wandb_entity_name YOUR_ENTITY to the training command"
