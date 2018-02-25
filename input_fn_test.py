import argparse
import logging
import numpy as np
import os
import tensorflow as tf 

from model.input_fn import input_fn
from model.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=os.path.join('experiments','base_model'),
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default=os.path.join('datasets','tensor'),
                    help="Directory containing the dataset")

if __name__ == '__main__':
    
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    # # Set the logger
    # set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, 'train')

    train_filenames = os.listdir(train_data_dir)
    train_features_data =  [os.path.join(train_data_dir, f) for f in train_filenames if f == 'X.npy'][0]
    train_labels_data = [os.path.join(train_data_dir, f) for f in train_filenames if f == "Y.npy"][0]

    train_inputs = input_fn(train_features_data, train_labels_data, params)

    print(train_inputs)