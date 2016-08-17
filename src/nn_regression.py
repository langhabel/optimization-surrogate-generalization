"""A neural network implementation for of nonlinear regression in TensorFlow."""
import numpy as np
import tensorflow as tf
import math

class NN_Regression:
    """A neural network implementation for nonlinear regression in TensorFlow.

    To operate use methods fit(X, Y) and predict(X).

    Includes implementation of Gal and Ghahramani's MC dropout uncertainty estimation.
    """

    def __init__(self,
                 initial_learning_rate=0.01,
                 dropout_rate_train=1.0,
                 hidden_layer_size=200,
                 hidden_layers=2,
                 batch_size=None,
                 method='Adam',
                 decay_rate=0.99,
                 decay_freq=1000,
                 regularization_factor=0.0,
                 data_dim=1,
                 weight_init_factor=1.0,
                 display_step=500):
        """Sets up the TensorFlow graph and initializes class variables.

        Args:
            initial_learning_rate: the initial learning rate for the neural net optimizer
            dropout_rate_train: the probability of units being kept in dropout training; 1.0 means no dropout is used
            hidden_layer_size: number of hidden units per hidden layer
            hidden_layers: number of hidden layers; must be either 1 or 2
            batch_size: training batch size. It must divide number of points given; None means use the whole batch
            method: optimization algorithm used; Adam or GradientDescent
            decay_rate: optimization decay rate (only for GradientDescent optimizer)
            decay_freq: optimization rate decay frequency given in epochs (only for GradientDescent optimizer)
            regularization_factor: the amount of L2-norm weight decay regularization
            data_dim: the dimensionality of the input data
            weight_init_factor: multiplicative factor to the randomly initialized weights and biases
            display_step: the interval for prints of training status
        """

        self.display_step = display_step
        self.dropout_rate_train = dropout_rate_train
        self.batch_size = batch_size
        self.weight_init_factor = weight_init_factor
        self.data_dim = data_dim

        ###---SETUP---###
        self.sess = tf.Session()

        # initialize Placeholders
        self.X_pl = tf.placeholder(tf.float32, shape=(None, data_dim), name='x_input')
        self.Y_pl = tf.placeholder(tf.float32, shape=(None, 1), name='y_label')
        self.keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob')
        self.data_len_pl = tf.placeholder(tf.float32, name='data_length')

        # build tensor graph
        hidden1, self.weight_h1, self.bias_h1 = self._nn_layer(self.X_pl, data_dim, hidden_layer_size)
        dropped1 = tf.nn.dropout(hidden1, self.keep_prob_pl)

        if hidden_layers == 2:
            hidden2, self.weight_h2, self.bias_h2 = self._nn_layer(dropped1, hidden_layer_size, hidden_layer_size, act=tf.sigmoid)
            dropped2 = tf.nn.dropout(hidden2, self.keep_prob_pl)
            self.output, self.weight_out, self.bias_out = self._regression_output_layer(dropped2, hidden_layer_size)
        elif hidden_layers == 1:
            self.output, self.weight_out, self.bias_out = self._regression_output_layer(dropped1, hidden_layer_size)
        else:
            raise ValueError('Can be either 1 or 2 hidden layers.')

        # loss and training operands
        self.l2loss = tf.nn.l2_loss(self.output - self.Y_pl) / self.data_len_pl

        # regularization
        regularizers = tf.nn.l2_loss(self.weight_h1) + tf.nn.l2_loss(self.bias_h1) + tf.nn.l2_loss(self.weight_out) + tf.nn.l2_loss(self.bias_out)
        if hidden_layers == 2:
            regularizers += tf.nn.l2_loss(self.weight_h2) + tf.nn.l2_loss(self.bias_h2)

        self.cost = self.l2loss + regularization_factor * regularizers

        if method == 'Adam':
            self.optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate).minimize(self.cost)
        elif method == 'GradientDescent':
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_freq, decay_rate, staircase=True)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost, global_step=global_step)
        else:
            raise ValueError('Can be either Adam or GradientDescent optimizer.')

        init = tf.initialize_all_variables()
        self.sess.run(init)


    ###---NN-BUILDING-BLOCKS---###
    def _nn_layer(self, input_tensor, input_dim, output_dim, act=tf.nn.relu):
        """Sets up a hidden layer and returns the respective node and weight variables."""
        weights = tf.Variable(self.weight_init_factor * tf.truncated_normal(shape=[input_dim, output_dim],
                                                                            stddev=math.sqrt(2.0)))
        biases = tf.Variable(self.weight_init_factor * tf.truncated_normal(shape=[output_dim],
                                                                           stddev=math.sqrt(2.0)))
        preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, 'activation')
        return activations, weights, biases

    def _regression_output_layer(self, input_tensor, input_dim):
        """Sets up the output layer and returns the respective node and weight variables."""
        weights = tf.Variable(self.weight_init_factor * tf.truncated_normal(shape=[input_dim, 1],
                                                                            stddev=math.sqrt(2.0)))
        biases = tf.Variable(self.weight_init_factor * tf.truncated_normal(shape=[1],
                                                                           stddev=math.sqrt(2.0)))
        activation = tf.matmul(input_tensor, weights) + biases
        return activation, weights, biases


    ###---FUNCTIONS---###
    def _nr_batches(self, data_size):
        """Computes the number of batches according to the given batch size and size of the dataset."""
        batch_size = self.batch_size
        if batch_size == None or batch_size > data_size:
            batch_size = data_size
        nr_batches = data_size / batch_size
        return nr_batches, batch_size

    def _get_batch(self, X, Y, step):
        """Returns the batch for a given training step."""
        nr_batches, batch_size = self._nr_batches(X.shape[0])

        begin = (step % nr_batches) * batch_size
        end = begin + batch_size

        X_batch = X[begin:end]
        if Y is None:
            Y_batch = None
        else:
            Y_batch = Y[begin:end]

        return X_batch, Y_batch

    def _compute_cost(self, X, Y):
        """Computes the prediction cost for the given data."""
        nr_batches, batch_size = self._nr_batches(X.shape[0])

        cost_arr = np.zeros((nr_batches))
        for epoch in range(nr_batches):
            X_batch, Y_batch = self._get_batch(X, Y, epoch)
            cost_arr[epoch] = self._compute_batch_cost(X_batch, Y_batch)
        return np.sum(cost_arr) / nr_batches


    def _compute_batch_cost(self, X_batch, Y_batch):
        """Computes the cost for one batch."""
        cost_arr = self.sess.run(self.l2loss, feed_dict={self.X_pl: X_batch,
                                                        self.Y_pl: Y_batch,
                                                        self.keep_prob_pl: 1.0,
                                                        self.data_len_pl: X_batch.shape[0]})
        return cost_arr  # one float per batch

    def _run_training(self, X, Y, training_epochs, verbose):
        """Trains the network for the given number of epochs."""
        for epoch in range(training_epochs):
            X_batch, Y_batch = self._get_batch(X, Y, epoch)

            indices = np.arange(X_batch.shape[0])
            np.random.shuffle(indices)
            X_batch = X_batch[indices]
            Y_batch = Y_batch[indices]


            _ = self.sess.run(self.optimizer, feed_dict={self.X_pl: X_batch,
                                                        self.Y_pl: Y_batch,
                                                        self.keep_prob_pl: self.dropout_rate_train,
                                                        self.data_len_pl: X_batch.shape[0]})

            # display training error
            if epoch % self.display_step == 0 and verbose:
                cost_arr = self._compute_batch_cost(X_batch, Y_batch)
                print('Epoch %04d: error=%.9f' % (epoch+1, cost_arr))

        cost = self._compute_cost(X, Y)
        print('Neurotic Natalie tamed - Training error=%.9f' % cost)
        return cost

    def _run_testing(self, X, Y):
        """Compute the cost on given test data."""
        cost = self._compute_cost(X, Y)
        print('Test error=%.9f' % cost)
        return cost

    def _standard_predict(self, X):
        """Returns the predictions for a given set of test points."""
        nr_batches, batch_size = self._nr_batches(X.shape[0])

        estimates = np.zeros((nr_batches, batch_size))
        for i in range(nr_batches):
            X_batch, _ = self._get_batch(X, None, i)

            # output returns array of batch_size floats per batch
            estimate = self.sess.run(self.output, feed_dict={self.X_pl: X_batch,
                                                             self.keep_prob_pl: 1.0,
                                                             self.data_len_pl: batch_size})
            estimates[i,:] = estimate.flatten()
        return estimates.flatten()

    def _dropout_predict(self, X, dropout_rate, T):
        """Gathers monte-carlo samples for MC dropout."""
        estimates = np.zeros((T, X.shape[0]))
        for i in range(T):
            # don't use batches but full test set
            est_list = self.sess.run([self.output], feed_dict={ self.X_pl: X,
                                                                self.keep_prob_pl: dropout_rate,
                                                                self.data_len_pl: X.shape[0]})
            estimates[i, :] = np.array(est_list).flatten()
        return estimates


    def _preprocess_data(self, X, Y):
        """Aligns matrix/array shapes for the case of 1D."""
        if self.data_dim == 1:
            X = np.matrix(X).T
        if Y is not None:
            Y = np.matrix(Y).T
        return X, Y

    ###---INTERFACE---###
    def fit(self, X, Y, training_epochs=3000, verbose=False):
        """Trains the neural network, given the training data.

        Args:
            X: training data in shape (n_samples, dim)
            Y: training labels
            verbose: whether to print training error every once in a while

        Returns:
            the training error
        """
        X, Y = self._preprocess_data(X, Y)
        return self._run_training(X, Y, training_epochs, verbose)

    def test(self, X, Y):
        """Tests the fit of the trained network, given the test data.

        Args:
            X: test data in shape (n_samples, dim)
            Y: test labels

        Returns:
            the test error
        """
        X, Y = self._preprocess_data(X, Y)
        return self._run_testing(X, Y)

    def predict(self, X):
        """Returns the estimated labels for new data points.

        Args:
            X: new data points in shape (n_samples, dim)

        Returns:
            estimated labels
        """
        X, _ = self._preprocess_data(X, None)
        estimates = self._standard_predict(X)
        return estimates

    def predict_uncertainty(self, X, dropout_rate=0.995, T=1000):
        """Uses MC Dropout to for each input data point compute the predicted label as well as uncertainty.

        Args:
            X: new data points
            dropout_rate: percentage of units kept during test time
            T: number of Monte Carlo samples taken

        Returns:
            predicted labels and uncertainty
        """
        X, _ = self._preprocess_data(X, None)
        estimates = self._dropout_predict(X, dropout_rate, T)
        prediction = np.mean(estimates, axis=0)
        uncertainty = np.var(estimates, axis=0)
        return prediction, uncertainty