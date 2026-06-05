#!/bin/bash
git clone https://github.com/facebookresearch/seamless_interaction
cd seamless_interaction
pip install -e .

# Run the download script
python ../download_seamless.py
