""" Imports MATLAB .mat files with test, dev, and train data and 
parses them into NumPy arrays in the appropriate directories:

datasets/
    dev/
    test/
    train/

For the first round, the MATLAB .mat files have already been parsed 
into appropriate proportions for train/dev/test and imported separately.
A 80/10/10 split is used.  The dimensions currently are:

train: 
    X is a 241x24000 tensor containing 24K training examples w/ 241 features
    Y is a 6x24000 tensor containing 24K matching true outputs w/ 6 features
dev: 
    X is a 241x3000 tensor containing 3K training examples w/ 241 features
    Y is a 6x3000 tensor containing 3K matching true outputs w/ 6 features
test: 
    X is a 241x3000 tensor containing 3K training examples w/ 241 features
    Y is a 6x3000 tensor containing 3K matching true outputs w/ 6 features
"""

import argparse
import numpy as np
import os
import scipy.io as sio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.path.join('datasets','matlab'), help="Directory with IROD dataset .mat files")
parser.add_argument('--output_dir', default=os.path.join('datasets','tensor'), help="Directory with IROD dataset Numpy tensors")

def convert_and_save(filename, output_dir):
    """ Converts from MATLAB .mat file to NumPy array """
    mat_contents = sio.loadmat(filename)
    if 'X' in mat_contents:
        data = mat_contents['X']
    if 'Y' in mat_contents:
        data = mat_contents['Y']

    print("Input filename is " + filename)
    save_name = filename.split(str(os.sep))[-1]
    save_name = save_name.split('.')[0]
    np.save(os.path.normpath(os.path.join(output_dir, save_name)), data)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    train_data_dir = os.path.normpath(os.path.join(args.data_dir, 'train'))
    dev_data_dir = os.path.join(args.data_dir, 'dev')
    test_data_dir = os.path.join(args.data_dir, 'test')

    train_filenames = os.listdir(train_data_dir)
    train_filenames = [os.path.normpath(os.path.join(train_data_dir, f)) for f in train_filenames if f.endswith('.mat')]
    dev_filenames = os.listdir(dev_data_dir)
    dev_filenames = [os.path.normpath(os.path.join(dev_data_dir, f)) for f in dev_filenames if f.endswith('.mat')]
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.normpath(os.path.join(test_data_dir, f)) for f in test_filenames if f.endswith('.mat')]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}
        
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(os.path.normpath(args.output_dir)))

    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.normpath(os.path.join(args.output_dir, '{}'.format(split)))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            convert_and_save(filename, output_dir_split)

    print("Done building dataset")