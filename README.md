# Moirain

This is the code repository for the paper Multimodal Alignment and Preference Optimization for Zero-Shot Conditional RNA Generation. It contains scripts for training and inference of the Moirain suite of models, enabling the generation of RNA sequences that bind to a user-specified input protein. This repository and the released model weights are the ones used in the paper.

## Installation
Run the following to create a conda environment with the necessary dependencies.
```bash
conda create -n moirain python=3.10
```
Next, after the activation of ```moirain``` environment, install required libraries.
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pandas
pip install hydra-core --upgrade
pip install einops
pip install transformers
pip install gemmi
pip install h5py
pip install peft
```
We recommend installing our code as a package. To do this, run the following.
```bash
pip install -e .
```


## Inference

To run Moirain-Multi or Moirain-DPO inference for protein of your choice use the following command:
```bash
python experiments/inference_moirain_dpo_from_cif.py --data.cif_path /path/to/your/name.cif
```
Please make sure that protein structure follows AlphaFold format. The command will generate 1000 RNA sequences and save them in the ```./inference_outputs/name``` folder. There will be two files: ```seq_na.fasta```, containing final RNA sequences, and ```seq_na_uncut.fasta```, containing corresponding tokens.