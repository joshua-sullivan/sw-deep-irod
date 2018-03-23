"""Define the model."""
import numpy as np
import tensorflow as tf 

def build_model(mode, inputs, params):
    feature = inputs['features']

    if params.model_version == 'lstm':

        # Multi-layer LSTM-based RNN
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params.lstm_num_units]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        lstm_output, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=feature, dtype=tf.float64)
        predictions = tf.layers.dense(lstm_output, params.output_length)


    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return predictions

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        predictions = build_model(is_training, inputs, params)

    # Define loss and accuracy
    loss = tf.losses.absolute_difference(labels=labels, predictions=predictions[:, params.num_meas - 1, :], weights=1.0e-03)
    accuracy = tf.losses.absolute_difference(labels=labels, predictions=predictions[:, params.num_meas - 1, :], weights=1.0e-03) 

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            # 'accuracy': tf.metrics.mean_squared_error(labels=labels, 
            #                                           predictions=predictions[:, 19, :], weights=1.0e-03),
            'accuracy': tf.metrics.mean(accuracy),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec