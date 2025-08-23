#!/usr/bin/env -S bash -i
echo "Creating conda environment"
conda create -n 07_xai -y python=3.11 
eval "$(conda shell.bash hook)"
conda activate 07_xai
# Check if the environment is activated
if [[ "$CONDA_DEFAULT_ENV" == "07_xai" ]]; then
    echo "Environment activated successfully for package installation"
else
    echo "Failed to activate environment for package installation. Dependencies not installed!"
    exit
fi

pip install uv
uv pip install -r requirements.txt

python -m ipykernel install --user --name "07_xai"

echo "Training classifier model"
python extras/train_classifier.py

conda deactivate
