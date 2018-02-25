"""Evaluate the model"""

import argparse
import logging
import numpy as np
import os
import tensorflow as tf 

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=os.path.join('experiments','base_model'),
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default=os.path.join('datasets','tensor'),
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)    

    # Load the params
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    # # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, 'test')

    test_filenames = os.listdir(test_data_dir)
    test_features_data =  [os.path.join(test_data_dir, f) for f in test_filenames if f == 'X.npy'][0]
    test_labels_data = [os.path.join(test_data_dir, f) for f in test_filenames if f == "Y.npy"][0]

    # specify the size of the evaluation set
    test_data_loaded = np.load(test_features_data, mmap_mode='r')
    params.eval_size = test_data_loaded.shape[0]
    
    test_inputs = input_fn(test_features_data, test_labels_data, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)