"""Train the model"""

import argparse
import logging
import os
import random

import numpy as np
import tensorflow as tf

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=os.path.join('experiments','base_model'),
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default=os.path.join('datasets','tensor'),
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwriting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwriting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")

    # Get the filenames from the train and dev sets
    train_filenames = os.listdir(train_data_dir)
    train_features_data =  [os.path.join(train_data_dir, f) for f in train_filenames if f == 'X.npy'][0]
    train_labels_data = [os.path.join(train_data_dir, f) for f in train_filenames if f == "Y.npy"][0]

    dev_filenames = os.listdir(dev_data_dir)
    dev_features_data =  [os.path.join(dev_data_dir, f) for f in dev_filenames if f == 'X.npy'][0]
    dev_labels_data = [os.path.join(dev_data_dir, f) for f in dev_filenames if f == "Y.npy"][0]

    # Specify the sizes of the dataset we train on and evaluate on
    train_data_loaded = np.load(train_features_data, mmap_mode='r')
    params.train_size = train_data_loaded.shape[0]

    dev_data_loaded = np.load(dev_features_data, mmap_mode='r')
    params.eval_size = dev_data_loaded.shape[0]

    # Create the two iterators over the two datasets
    train_inputs = input_fn(train_features_data, train_labels_data, params)
    dev_inputs = input_fn(dev_features_data, dev_labels_data, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    dev_model_spec = model_fn('eval', dev_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, dev_model_spec, args.model_dir, params, args.restore_from)
