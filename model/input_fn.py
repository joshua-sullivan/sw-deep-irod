"""Create the input data pipeline using `tf.data`"""

import numpy as np
import tensorflow as tf


def input_fn(features_in, labels_in, params):
    """Input function for the dataset.

    Args:
        features_in: path to NumPy array with m feature vector rows, where m is # of examples
        labels_in: path to Numpy array with m label vector rows
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    features = np.load(features_in)
    labels = np.load(labels_in)

    num_examples = features.shape[0]

    features = features.reshape(num_examples, 20, 12)

    assert features.shape[0] == labels.shape[0], "Feature tensor and output tensor should have same number of examples."

    dataset = (tf.data.Dataset.from_tensor_slices((features, labels))
        .batch(params.batch_size)
        .prefetch(1)
    )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    feature, label = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'features': feature, 
              'labels': label, 
              'iterator_init_op': iterator_init_op}
              
    return inputs
