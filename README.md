# Exploring the Potential of Unsupervised Representations of IR Satellite Images for Typhoon Forecasting 

### Dependencies
PyPhoon2: https://github.com/kitamoto-lab/pyphoon2
Digital Typhoon Dataset: https://github.com/kitamoto-lab/digital-typhoon

Requires to independtly install open-mpi (apt, brew or build from scratch)
See `/requirements.txt`

### Repository structure
`/configs`: contains 2 example configuration files
    - `moco_seq.conf` to train a Resnet34 model using MoCo
    - `train_lstm.conf` to train an LSTM model using sequences preprocessed by an image encoder trained using the above configuration

To automatically preprocess all sequences in the Digital Typhoon dataset, run the `create_preprocessed_sequences.py` file. You can modify parameters by modifying this file's main function.

The `/lib/utils/dataset.py` contains all important PyTorch dataset created to adapt the PyPhoon2 interface to this project's specific needs.

The `/lib/trainers` directory contains a `BaseTrainer` class and 2 different specialized trainers, one for MoCo, one for LSTM forecasting.

The `tests.ipynb` notebook contains the code needed to test the performance of the LSTM forecasting models as well as the code used to generate GIFs of forecasting inputs/outputs across sequences.

### Example GIF output for the 202302 Typhoon sequence

<p align="center"> <img src='gifs/202302.gif' align="center""> </p> 
