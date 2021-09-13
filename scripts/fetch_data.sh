#!/bin/bash

# fetach data used in SPIN training and in TUCH training and evaluation.

# Pretrained SPIN checkpoint
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data
mv data/model_checkpoint.pt data/spin_model_checkpoint.pt

# Get the TUCH checkpoint
wget https://download.is.tue.mpg.de/tuch/tuch_model_checkpoint.pt --directory-prefix=data
