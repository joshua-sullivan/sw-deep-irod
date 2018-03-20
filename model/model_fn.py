"""Define the model."""
import numpy as np
import tensorflow as tf 

# def init_weight_and_bias(name, shape):
#     weight = tf.get_variable(name=name[0], shape=shape, dtype=tf.float64, 
#                              initializer=tf.contrib.layers.xavier_initializer(seed=1))

#     bias = tf.get_variable(name=name[1], shape=[shape[1]], dtype=tf.float64, 
#                            initializer=tf.zeros_initializer())

#     return weight, bias

# def create_fully_connected(input_layer, weights, biases):
#     layer = tf.add(tf.matmul(input_layer, weights), biases)

#     return layer


# def build_model(is_training, inputs, params):
#     features = inputs['features']
#     labels = inputs['labels'] 

#     (m, n_x) = features.shape

#     # print(features.shape)
#     n_y = labels.shape[1]

#     # Creating the first hidden layer with 30 nodes
#     num_nodes_1 = 300
#     weight_1, bias_1 = init_weight_and_bias(name=["W1", "b1"], shape=[n_x, num_nodes_1])
#     # print('First layer')
#     # print(weight_1.shape)
#     # print(bias_1.shape)
#     layer_1 = tf.nn.relu(create_fully_connected(features, weight_1, bias_1))
#     # print(layer_1.shape)
#     if params.use_batch_norm:
#         layer_1 = tf.layers.batch_normalization(layer_1, momentum=params.bn_momentum, 
#                                                          training=is_training)

#     # Creating the second hidden layer with 30 nodes
#     num_nodes_2 = 200
#     weight_2, bias_2 = init_weight_and_bias(name=["W2", "b2"], shape=[num_nodes_1, num_nodes_2])
#     # print('Second layer')
#     # print(weight_2.shape)
#     # print(bias_2.shape)
#     layer_2 = tf.nn.relu(create_fully_connected(layer_1, weight_2, bias_2))
#     # print(layer_2.shape)
#     if params.use_batch_norm:
#         layer_2 = tf.layers.batch_normalization(layer_2, momentum=params.bn_momentum, 
#                                                          training=is_training)

#     # Creating the third hidden layer with 10 nodes
#     num_nodes_3 = 100
#     weight_3, bias_3 = init_weight_and_bias(name=["W3", "b3"], shape=[num_nodes_2, num_nodes_3])
#     # print('Third layer')
#     # print(weight_3.shape)
#     # print(bias_3.shape)
#     layer_3 = tf.nn.relu(create_fully_connected(layer_2, weight_3, bias_3))
#     # print(layer_3.shape)
#     if params.use_batch_norm:
#         layer_3 = tf.layers.batch_normalization(layer_3, momentum=params.bn_momentum,
#                                                          training=is_training)

#     # Creating the output layer 
#     num_nodes_out = n_y
#     weight_out, bias_out = init_weight_and_bias(name=["Wout", "bout"], shape=[num_nodes_3, num_nodes_out])
#     # print('Output layer')
#     # print(weight_out.shape)
#     # print(bias_out.shape)
#     prediction = create_fully_connected(layer_3, weight_out, bias_out)
#     # print(prediction.shape)

#     return prediction    

def build_model(mode, inputs, params):
    feature = inputs['features']

    if params.model_version == 'lstm':

        # Multi-layer LSTM-based RNN
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in params.lstm_num_units]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        output, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=feature, dtype=tf.float64)
        predictions = tf.layers.dense(output, params.output_length)

        # # Single-layer LSTM-based RNN
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        # output, _ = tf.nn.dynamic_rnn(lstm_cell, feature, dtype=tf.float64)
        # predictions = tf.layers.dense(output, params.output_length)

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
    # loss = tf.losses.huber_loss(labels=labels, predictions=predictions[:, 19, :], weights=1.0, delta=5000.)
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions[:, 19, :], weights=1.0e-06) 
    loss = tf.losses.absolute_difference(labels=labels, predictions=predictions[:, 19, :], weights=1.0e-03)

    # res = labels - predictions
    # labels_mean = tf.reduce_mean(labels)
    # SStot = tf.reduce_sum(tf.squared_difference(labels, labels_mean))
    # SSreg = tf.reduce_sum(tf.squared_difference(predictions, labels_mean))
    # SSres = tf.reduce_sum(tf.squared_difference(labels, predictions))
    # R2 = 1 - (SSres/SStot)

    # accuracy = tf.reduce_mean(tf.squared_difference(predictions[:, 19, :], labels))
    accuracy = tf.losses.absolute_difference(labels=labels, predictions=predictions[:, 19, :], weights=1.0e-03) 
    # accuracy = R2

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        # if params.use_batch_norm:
        #     # Add a dependency to update the moving mean and variance for batch normalization
        #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #         train_op = optimizer.minimize(loss, global_step=global_step)
        # else:
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