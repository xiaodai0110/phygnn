# -*- coding: utf-8 -*-
"""
Physics Guided Neural Network
"""

import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import losses, optimizers

from phygnn.base import CustomNetwork
from phygnn.utilities.loss_metrics import METRICS

logger = logging.getLogger(__name__)


class PhysicsGuidedNeuralNetwork(CustomNetwork):
    """Simple Deep Neural Network with custom physical loss function.

    Note that the phygnn model requires TensorFlow 2.x
    """

    def __init__(
        self,
        p_fun,
        loss_weights=(0.5, 0.5),
        n_features=1,
        n_labels=1,
        hidden_layers=None,
        input_layer=None,
        output_layer=None,
        layers_obj=None,
        metric='mae',
        optimizer=None,
        learning_rate=0.01,
        history=None,
        kernel_reg_rate=0.0,
        kernel_reg_power=1,
        bias_reg_rate=0.0,
        bias_reg_power=1,
        baseline_init=None,
        baseline_mu=None,
        baseline_stdev=None,
        baseline_trainable=False,
        baseline_loss_weight=1.0,
        feature_names=None,
        output_names=None,
        name=None,
    ):
        """
        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs. y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        n_features : int, optional
            Number of input features. This should match the last dimension
            of the feature training data.
        n_labels : int, optional
            Number of output labels. This should match the last dimension
            of the label training data.
        hidden_layers : list, optional
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 8 hidden layers (10 layers including input+output):
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01},
                 {'class': 'Flatten'},
                 ]
        input_layer : None | bool | dict
            Input layer. specification. Can be a dictionary similar to
            hidden_layers specifying a dense / conv / lstm layer.  Will
            default to a keras InputLayer with input shape = n_features.
            Can be False if the input layer will be included in the
            hidden_layers input.
        output_layer : None | bool | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}].
            This defaults to a single dense layer with no activation
            (best for regression problems).  Can be False if the output layer
            will be included in the hidden_layers input.
        layers_obj : None | phygnn.utilities.tf_layers.Layers
            Optional initialized Layers object to set as the model layers
            including pre-set weights. This option will override the
            hidden_layers, input_layer, and output_layer arguments.
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
            or a method in tensorflow.keras.losses that takes
            (y_true, y_predicted) as arguments.
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        history : None | pd.DataFrame, optional
            Learning history if continuing a training session.
        kernel_reg_rate : float, optional
            Kernel regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer weights and should reduce
            model complexity. Setting this to 0.0 will disable kernel
            regularization.
        kernel_reg_power : int, optional
            Kernel regularization power. kernel_reg_power=1 is L1
            regularization (lasso regression), and kernel_reg_power=2 is L2
            regularization (ridge regression).
        bias_reg_rate : float, optional
            Bias regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer biases and should reduce
            model complexity. Setting this to 0.0 will disable bias
            regularization.
        bias_reg_power : int, optional
            Bias regularization power. bias_reg_power=1 is L1
            regularization (lasso regression), and bias_reg_power=2 is L2
            regularization (ridge regression).
        baseline_init : float | None, optional
            Initial value for a residual baseline k0. When set, the network
            output is interpreted as a residual that can be added to this
            baseline with ``predict_k``.
        baseline_mu : float | None, optional
            Experimental prior mean for k0. Defaults to baseline_init.
        baseline_stdev : float | None, optional
            Experimental standard deviation for the k0 prior loss.
        baseline_trainable : bool, optional
            Flag to make k0 a trainable scalar variable.
        baseline_loss_weight : float, optional
            Default loss weight for the k0 prior in multi-fidelity training.
        feature_names : list | tuple | None, optional
            Training feature names (strings). Mostly a convenience so that a
            loaded-from-disk model will have declared feature names, making it
            easier to feed in features for prediction. This will also get set
            if phygnn is trained on a DataFrame.
        output_names : list | tuple | None, optional
            Prediction output names (strings). Mostly a convenience so that a
            loaded-from-disk model will have declared output names, making it
            easier to understand prediction output. This will also get set
            if phygnn is trained on a DataFrame.
        name : None | str
            Optional model name for debugging.
        """

        super().__init__(
            n_features=n_features,
            n_labels=n_labels,
            hidden_layers=hidden_layers,
            input_layer=input_layer,
            output_layer=output_layer,
            layers_obj=layers_obj,
            feature_names=feature_names,
            output_names=output_names,
        )

        self._p_fun = p_fun if p_fun is not None else self.p_fun_dummy
        self._loss_weights = None
        self._metric = metric
        self._optimizer = None
        self._history = history
        self._learning_rate = learning_rate
        self.kernel_reg_rate = kernel_reg_rate
        self.kernel_reg_power = kernel_reg_power
        self.bias_reg_rate = bias_reg_rate
        self.bias_reg_power = bias_reg_power
        self._k0 = None
        self._baseline_mu = None
        self._baseline_stdev = None
        self._baseline_trainable = bool(baseline_trainable)
        self._baseline_loss_weight = baseline_loss_weight
        self.name = name if isinstance(name, str) else 'phygnn'

        if baseline_init is None and baseline_mu is not None:
            baseline_init = baseline_mu
        if baseline_init is not None:
            self.set_baseline(
                baseline_init,
                mu=baseline_mu,
                stdev=baseline_stdev,
                trainable=baseline_trainable,
            )

        self.set_loss_weights(loss_weights)

        if self._metric.lower() in METRICS:
            self._metric_fun = METRICS[self._metric.lower()]
        else:
            try:
                self._metric_fun = getattr(losses, self._metric)
            except Exception as e:
                msg = (
                    'Could not recognize error metric "{}". The following '
                    'error metrics are available: {}'.format(
                        self._metric, list(METRICS.keys())
                    )
                )
                logger.error(msg)
                raise KeyError(msg) from e

        self._optimizer = optimizer
        if isinstance(optimizer, dict):
            if 'class_name' not in optimizer:
                optimizer = {
                    'class_name': optimizer['name'],
                    'config': optimizer,
                }
            self._optimizer = optimizers.deserialize(optimizer)
        elif optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)

    @staticmethod
    def p_fun_dummy(model, y_true, y_predicted, p):  # noqa : ARG004
        """Example dummy function for p loss calculation.

        This dummy function does not do a real physics calculation, it just
        shows the required p_fun interface and calculates a normal MAE loss
        based on y_predicted and y_true.

        Parameters
        ----------
        model : PhysicsGuidedNeuralNetwork
            Instance of the phygnn model at the current point in training.
        y_true : np.ndarray
            Known y values that were given to the phygnn.fit() method.
        y_predicted : tf.Tensor
            Predicted y values in a >=2D tensor based on x values in the
            current batch.
        p : np.ndarray
            Supplemental physical feature data that can be used to calculate a
            y_physical value to compare against y_predicted. The rows in this
            array have been carried through the batching process alongside
            y_true and the x-features used to create y_predicted and so can be
            used 1-to-1 with the rows in y_predicted and y_true.

        Returns
        -------
        p_loss : tf.Tensor
            A 0D tensor physical loss value.
        """
        # pylint: disable=W0613
        return tf.math.reduce_mean(tf.math.abs(y_predicted - y_true))

    def preflight_data(self, x, y, p):
        """Run simple preflight checks on data shapes and data types.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. Generally speaking, the data should
            always have the number of observations in the first axis and the
            number of features/channels in the last axis. Spatial and temporal
            dimensions can be used in intermediate axes.
        y : np.ndarray | pd.DataFrame
            Known output data in a >=2D array or DataFrame.
            Same dimension rules as x.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in >=2D
            array or DataFrame. Same dimension rules as x.

        Returns
        -------
        x : np.ndarray
            Feature data
        y : np.ndarray
            Known output data
        p : np.ndarray
            Supplemental feature data
        """

        self._check_shapes(x, y)
        self._check_shapes(x, p)

        if self._n_features is None:
            self._n_features = x.shape[-1]
        if self._n_labels is None:
            self._n_labels = y.shape[-1]

        x_msg = 'x data has {} features but expected {}'.format(
            x.shape[-1], self._n_features
        )
        y_msg = 'y data has {} features but expected {}'.format(
            y.shape[-1], self._n_labels
        )
        assert x.shape[-1] == self._n_features, x_msg
        assert y.shape[-1] == self._n_labels, y_msg

        x = self.preflight_features(x)

        if isinstance(y, pd.DataFrame):
            y_cols = y.columns.values.tolist()
            if self.output_names is None:
                self.output_names = y_cols
            else:
                msg = (
                    'Cannot work with input y columns: {}, previously set '
                    'output names are: {}'.format(y_cols, self.output_names)
                )
                assert self.output_names == y_cols, msg
            y = y.values

        if isinstance(p, pd.DataFrame):
            p = p.values

        return x, y, p

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self._history

    @property
    def weights(self):
        """
        Get model weights for gradient calculations.

        This includes the optional residual baseline k0 when it is trainable.

        Returns
        -------
        list
        """
        weights = super().weights
        if self._k0 is not None and self.baseline_trainable:
            weights = weights + [self._k0]

        return weights

    @property
    def baseline(self):
        """
        Current residual baseline k0 value.

        Returns
        -------
        float | None
        """
        if self._k0 is None:
            return None

        return float(self._k0.numpy())

    @property
    def baseline_mu(self):
        """
        Experimental prior mean for k0.

        Returns
        -------
        float | None
        """
        return self._baseline_mu

    @property
    def baseline_stdev(self):
        """
        Experimental prior standard deviation for k0.

        Returns
        -------
        float | None
        """
        return self._baseline_stdev

    @property
    def baseline_trainable(self):
        """
        Flag indicating whether k0 is trainable.

        Returns
        -------
        bool
        """
        return self._baseline_trainable

    @property
    def baseline_loss_weight(self):
        """
        Default loss weight for the k0 prior loss.

        Returns
        -------
        float
        """
        return self._baseline_loss_weight

    def set_baseline(self, value, mu=None, stdev=None, trainable=None):
        """Set the residual baseline k0 and its experimental prior.

        Parameters
        ----------
        value : float
            Current/initial value for k0.
        mu : float | None, optional
            Experimental prior mean. Defaults to value.
        stdev : float | None, optional
            Experimental standard deviation for the prior loss.
        trainable : bool | None, optional
            Flag to make k0 trainable. None preserves the current setting.
        """
        if trainable is None:
            trainable = self.baseline_trainable

        self._baseline_trainable = bool(trainable)
        self._baseline_mu = float(value if mu is None else mu)
        self._baseline_stdev = (
            None if stdev is None else float(stdev)
        )
        self._k0 = tf.Variable(
            float(value),
            trainable=self._baseline_trainable,
            dtype=tf.float32,
            name='k0',
        )

    def set_learning_rate(self, learning_rate):
        """Set the optimizer learning rate in-place."""
        self._learning_rate = learning_rate
        try:
            self._optimizer.learning_rate.assign(learning_rate)
        except AttributeError:
            self._optimizer.learning_rate = learning_rate

    def predict_delta(self, x, **kwargs):
        """Predict the residual network output Delta k."""
        return self.predict(x, **kwargs)

    def predict_k(self, x, to_numpy=True, training=False):
        """Predict total k as k0 + Delta k.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data.
        to_numpy : bool
            Flag to convert output from tensor to numpy array.
        training : bool
            Flag for predict() used in the training routine.

        Returns
        -------
        tf.Tensor | np.ndarray
            Total property prediction.
        """
        if self._k0 is None:
            msg = (
                'Cannot calculate k0 + Delta k because no residual baseline '
                'has been set.'
            )
            logger.error(msg)
            raise RuntimeError(msg)

        y = self.predict(x, to_numpy=False, training=training)
        y = y + self._k0

        if to_numpy:
            y = y.numpy()

        return y

    @property
    def kernel_reg_term(self):
        """Get the regularization term for the kernel regularization without
        the regularization rate applied."""
        loss_k_reg = [tf.math.abs(x) for x in self.kernel_weights]
        loss_k_reg = [
            tf.math.pow(x, self.kernel_reg_power) for x in loss_k_reg
        ]
        loss_k_reg = tf.math.reduce_sum([
            tf.math.reduce_sum(x) for x in loss_k_reg
        ])

        return loss_k_reg

    @property
    def bias_reg_term(self):
        """Get the regularization term for the bias regularization without
        the regularization rate applied."""
        loss_b_reg = [tf.math.abs(x) for x in self.bias_weights]
        loss_b_reg = [tf.math.pow(x, self.bias_reg_power) for x in loss_b_reg]
        loss_b_reg = tf.math.reduce_sum([
            tf.math.reduce_sum(x) for x in loss_b_reg
        ])

        return loss_b_reg

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """

        model_params = super().model_params
        model_params.update({
            'p_fun': self._p_fun,
            'loss_weights': self._loss_weights,
            'metric': self._metric,
            'optimizer': optimizers.serialize(self._optimizer),
            'learning_rate': self._learning_rate,
            'layers_obj': self.layers_obj,
            'history': self.history,
            'kernel_reg_rate': self.kernel_reg_rate,
            'kernel_reg_power': self.kernel_reg_power,
            'bias_reg_rate': self.bias_reg_rate,
            'bias_reg_power': self.bias_reg_power,
            'baseline_init': self.baseline,
            'baseline_mu': self.baseline_mu,
            'baseline_stdev': self.baseline_stdev,
            'baseline_trainable': self.baseline_trainable,
            'baseline_loss_weight': self.baseline_loss_weight,
        })

        return model_params

    def preflight_p_fun(self, x, y_true, p, p_kwargs):
        """Run a pre-flight check making sure the p_fun is differentiable."""

        if p_kwargs is None:
            p_kwargs = {}

        with tf.GradientTape() as tape:
            y_predicted = self.predict(x, to_numpy=False)
            p_loss = self._p_fun(self, y_true, y_predicted, p, **p_kwargs)
            grad = tape.gradient(p_loss, self.weights)

            if not tf.is_tensor(p_loss):
                emsg = 'Loss output from p_fun() must be a tensor!'
                logger.error(emsg)
                raise TypeError(emsg)

            if p_loss.ndim > 1:
                emsg = (
                    'Loss output from p_fun() should be a scalar tensor '
                    'but received a tensor with shape {}'.format(p_loss.shape)
                )
                logger.error(emsg)
                raise ValueError(emsg)

            assert isinstance(grad, list)
            if grad[0] is None:
                emsg = (
                    'The input p_fun was not differentiable! '
                    'Please use only tensor math in the p_fun.'
                )
                logger.error(emsg)
                raise RuntimeError(emsg)

        logger.debug('p_fun passed preflight check.')

    def reset_history(self):
        """Erase previous training history without resetting trained weights"""
        self._history = None

    def set_loss_weights(self, loss_weights):
        """Set new loss weights

        Parameters
        ----------
        loss_weights : tuple
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        """
        assert np.sum(loss_weights) > 0, 'Sum of loss_weights must be > 0!'
        assert len(loss_weights) == 2, 'loss_weights can only have two values!'
        self._loss_weights = loss_weights

    def _parse_multifidelity_loss_weights(self, loss_weights):
        """Parse LF/HF/baseline loss weights."""
        out = {
            'lf': 1.0,
            'hf': 10.0,
            'baseline': self.baseline_loss_weight,
        }

        if loss_weights is None:
            return out

        if isinstance(loss_weights, dict):
            aliases = {
                'lambda_lf': 'lf',
                'lambda_hf': 'hf',
                'lambda_bn': 'baseline',
                'bn': 'baseline',
                'k0': 'baseline',
            }
            for key, value in loss_weights.items():
                key = aliases.get(key.lower(), key.lower())
                if key not in out:
                    msg = (
                        'Unrecognized multi-fidelity loss weight "{}". '
                        'Expected keys include lf, hf, and baseline.'
                    ).format(key)
                    logger.error(msg)
                    raise KeyError(msg)
                out[key] = float(value)

        elif isinstance(loss_weights, (tuple, list)):
            if len(loss_weights) not in {2, 3}:
                msg = (
                    'multi-fidelity loss_weights must have 2 or 3 values: '
                    '(lambda_lf, lambda_hf[, lambda_bn])'
                )
                logger.error(msg)
                raise ValueError(msg)
            out['lf'] = float(loss_weights[0])
            out['hf'] = float(loss_weights[1])
            if len(loss_weights) == 3:
                out['baseline'] = float(loss_weights[2])

        else:
            msg = (
                'multi-fidelity loss_weights must be a dict, tuple, list, '
                'or None, but received {}'.format(type(loss_weights))
            )
            logger.error(msg)
            raise TypeError(msg)

        msg = 'Sum of multi-fidelity loss weights must be > 0!'
        assert sum(out.values()) > 0, msg

        return out

    @staticmethod
    def _parse_label_type(label_type):
        """Parse the supervised target type."""
        label_type = label_type.lower()
        if label_type not in {'k', 'delta'}:
            msg = 'label_type must be either "k" or "delta".'
            logger.error(msg)
            raise ValueError(msg)

        return label_type

    def _predict_target(self, x, label_type='k', training=False):
        """Predict either total k or residual Delta k for training."""
        label_type = self._parse_label_type(label_type)
        if label_type == 'k':
            return self.predict_k(x, to_numpy=False, training=training)

        return self.predict(x, to_numpy=False, training=training)

    def calc_baseline_loss(self, mu=None, stdev=None, epsilon=1e-6):
        """Calculate the BN baseline prior loss for k0.

        Parameters
        ----------
        mu : float | None, optional
            Prior mean. Defaults to the stored baseline_mu.
        stdev : float | None, optional
            Prior standard deviation. Defaults to the stored baseline_stdev.
        epsilon : float, optional
            Small denominator stabilizer.

        Returns
        -------
        tf.Tensor
            Scalar prior loss ``((k0 - mu) / (stdev + epsilon)) ** 2``.
        """
        if self._k0 is None:
            return tf.constant(0.0, dtype=tf.float32)

        if mu is None:
            mu = self.baseline_mu
        if stdev is None:
            stdev = self.baseline_stdev

        if mu is None:
            return tf.constant(0.0, dtype=tf.float32)

        stdev = 0.0 if stdev is None else stdev
        mu = tf.constant(float(mu), dtype=tf.float32)
        denom = tf.constant(float(stdev) + epsilon, dtype=tf.float32)

        return tf.square((self._k0 - mu) / denom)

    def _regularization_loss(self):
        """Calculate structural regularization loss."""
        loss = tf.constant(0.0, dtype=tf.float32)

        if self.kernel_reg_rate != 0:
            loss += self.kernel_reg_term * self.kernel_reg_rate

        if self.bias_reg_rate != 0:
            loss += self.bias_reg_term * self.bias_reg_rate

        return loss

    def calc_multifidelity_loss(
        self,
        x_lf=None,
        y_lf=None,
        x_hf=None,
        y_hf=None,
        loss_weights=None,
        label_type='k',
        baseline_mu=None,
        baseline_stdev=None,
        baseline_epsilon=1e-6,
        training=False,
    ):
        """Calculate LF/HF residual PGNN loss.

        Parameters
        ----------
        x_lf : np.ndarray | None
            Low-fidelity features.
        y_lf : np.ndarray | None
            Low-fidelity labels. Use total k when label_type='k', or Delta k
            when label_type='delta'.
        x_hf : np.ndarray | None
            High-fidelity features.
        y_hf : np.ndarray | None
            High-fidelity labels. Use total k when label_type='k', or Delta k
            when label_type='delta'.
        loss_weights : dict | tuple | list | None
            LF/HF/baseline weights. Dict keys may include ``lf``, ``hf``,
            ``baseline``, ``lambda_lf``, ``lambda_hf``, or ``lambda_bn``.
        label_type : {'k', 'delta'}, optional
            Whether labels are total k or residual Delta k.
        baseline_mu : float | None, optional
            Baseline prior mean for this loss calculation.
        baseline_stdev : float | None, optional
            Baseline prior standard deviation for this loss calculation.
        baseline_epsilon : float, optional
            Small denominator stabilizer for the baseline prior.
        training : bool, optional
            Forward-pass training flag.

        Returns
        -------
        tuple
            ``(loss, lf_loss, hf_loss, baseline_loss, regularization_loss)``.
        """
        loss_weights = self._parse_multifidelity_loss_weights(loss_weights)

        loss = tf.constant(0.0, dtype=tf.float32)
        lf_loss = tf.constant(0.0, dtype=tf.float32)
        hf_loss = tf.constant(0.0, dtype=tf.float32)
        baseline_loss = tf.constant(0.0, dtype=tf.float32)

        if x_lf is not None and y_lf is not None and loss_weights['lf'] != 0:
            y_lf = tf.convert_to_tensor(y_lf, dtype=tf.float32)
            y_lf_predicted = self._predict_target(
                x_lf, label_type=label_type, training=training
            )
            lf_loss = self._metric_fun(y_lf, y_lf_predicted)
            msg = 'Bad shape from LF loss! Must be 0D but received: {}'
            assert lf_loss.ndim == 0, msg.format(lf_loss)
            loss += loss_weights['lf'] * lf_loss

        if x_hf is not None and y_hf is not None and loss_weights['hf'] != 0:
            y_hf = tf.convert_to_tensor(y_hf, dtype=tf.float32)
            y_hf_predicted = self._predict_target(
                x_hf, label_type=label_type, training=training
            )
            hf_loss = self._metric_fun(y_hf, y_hf_predicted)
            msg = 'Bad shape from HF loss! Must be 0D but received: {}'
            assert hf_loss.ndim == 0, msg.format(hf_loss)
            loss += loss_weights['hf'] * hf_loss

        if loss_weights['baseline'] != 0:
            baseline_loss = self.calc_baseline_loss(
                mu=baseline_mu,
                stdev=baseline_stdev,
                epsilon=baseline_epsilon,
            )
            msg = 'Bad shape from baseline loss! Must be 0D but received: {}'
            assert baseline_loss.ndim == 0, msg.format(baseline_loss)
            loss += loss_weights['baseline'] * baseline_loss

        reg_loss = self._regularization_loss()
        loss += reg_loss

        if tf.math.is_nan(loss):
            msg = 'phygnn calculated a NaN multi-fidelity loss value!'
            logger.error(msg)
            raise ArithmeticError(msg)

        return loss, lf_loss, hf_loss, baseline_loss, reg_loss

    def calc_loss(self, y_true, y_predicted, p, p_kwargs):
        """Calculate the loss function by comparing y_true to model-predicted y

        Parameters
        ----------
        y_true : np.ndarray
            Known output data in a >=2D array.
        y_predicted : tf.Tensor
            Model-predicted output data in a >=2D tensor.
        p : np.ndarray
            Supplemental feature data for the physics loss function in >=2D
            array
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.

        Returns
        -------
        loss : tf.tensor
            Sum of the NN loss function comparing the y_predicted against
            y_true and the physical loss function (self._p_fun) with
            respective weights applied.
        nn_loss : tf.tensor
            Standard NN training loss comparing y to y_predicted.
        p_loss : tf.tensor
            Physics loss from p_fun.
        """

        if p_kwargs is None:
            p_kwargs = {}

        loss = tf.constant(0.0, dtype=tf.float32)
        nn_loss = tf.constant(0.0, dtype=tf.float32)
        p_loss = tf.constant(0.0, dtype=tf.float32)

        if self._loss_weights[0] != 0:
            nn_loss = self._metric_fun(y_true, y_predicted)
            msg = 'Bad shape from nn_loss fun! Must be 0D but received: {}'
            msg = msg.format(nn_loss)
            assert nn_loss.ndim == 0, msg
            loss += self._loss_weights[0] * nn_loss

        if self._loss_weights[1] != 0:
            p_loss = self._p_fun(self, y_true, y_predicted, p, **p_kwargs)
            msg = 'Bad shape from p_loss fun! Must be 0D but received: {}'
            msg = msg.format(p_loss)
            assert p_loss.ndim == 0, msg
            loss += self._loss_weights[1] * p_loss

        logger.debug(
            'NN Loss: {:.2e}, P Loss: {:.2e}, Total Loss: {:.2e}'.format(
                nn_loss, p_loss, loss
            )
        )

        if self.kernel_reg_rate != 0:
            loss_kernel_reg = self.kernel_reg_term * self.kernel_reg_rate
            loss += loss_kernel_reg
            logger.debug(
                'Kernel regularization loss: {:.2e}, '
                'Total Loss: {:.2e}'.format(loss_kernel_reg, loss)
            )

        if self.bias_reg_rate != 0:
            loss_bias_reg = self.bias_reg_term * self.bias_reg_rate
            loss += loss_bias_reg
            logger.debug(
                'Bias regularization loss: {:.2e}, Total Loss: {:.2e}'.format(
                    loss_bias_reg, loss
                )
            )

        if tf.math.is_nan(loss):
            msg = 'phygnn calculated a NaN loss value!'
            logger.error(msg)
            raise ArithmeticError(msg)

        return loss, nn_loss, p_loss

    def _get_grad(self, x, y_true, p, p_kwargs):
        """Get the gradient based on a mini-batch of x and y_true data."""
        with tf.GradientTape() as tape:
            y_predicted = self.predict(x, to_numpy=False, training=True)
            loss, nn_loss, p_loss = self.calc_loss(
                y_true, y_predicted, p, p_kwargs
            )
            grad = tape.gradient(loss, self.weights)

        return grad, loss, nn_loss, p_loss

    def run_gradient_descent(self, x, y_true, p, p_kwargs):
        """Run gradient descent for one mini-batch of (x, y_true)
        and adjust NN weights."""
        grad, loss, nn_loss, p_loss = self._get_grad(x, y_true, p, p_kwargs)
        self._optimizer.apply_gradients(zip(grad, self.weights))
        return loss, nn_loss, p_loss

    def _preflight_output(self, y):
        """Run preflight checks on supervised output data."""
        if isinstance(y, pd.DataFrame):
            y_cols = y.columns.values.tolist()
            if self.output_names is None:
                self.output_names = y_cols
            else:
                msg = (
                    'Cannot work with input y columns: {}, previously set '
                    'output names are: {}'.format(y_cols, self.output_names)
                )
                assert self.output_names == y_cols, msg
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values

        y = np.asarray(y)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        if self._n_labels is None:
            self._n_labels = y.shape[-1]

        y_msg = 'y data has {} features but expected {}'.format(
            y.shape[-1], self._n_labels
        )
        assert y.shape[-1] == self._n_labels, y_msg

        return y

    def _preflight_optional_xy(self, x, y):
        """Run preflight checks on an optional supervised dataset."""
        if x is None and y is None:
            return None, None

        if x is None or y is None:
            msg = 'features and labels must both be provided or both be None.'
            logger.error(msg)
            raise ValueError(msg)

        y = self._preflight_output(y)
        self._check_shapes(x, y)
        x = self.preflight_features(x)

        return x, y

    def preflight_multifidelity_data(
        self, x_lf=None, y_lf=None, x_hf=None, y_hf=None
    ):
        """Run preflight checks on LF/HF supervised datasets."""
        x_lf, y_lf = self._preflight_optional_xy(x_lf, y_lf)
        x_hf, y_hf = self._preflight_optional_xy(x_hf, y_hf)

        if x_lf is None and x_hf is None:
            msg = 'At least one of LF or HF data must be supplied.'
            logger.error(msg)
            raise ValueError(msg)

        return x_lf, y_lf, x_hf, y_hf

    @classmethod
    def _train_val_split_xy(
        cls, x, y, shuffle=True, validation_split=0.2
    ):
        """Split an optional supervised dataset into train and validation."""
        if x is None:
            return (None, None), (None, None)

        if validation_split <= 0 or len(x) < 2:
            return (x, y), (None, None)

        L = x.shape[0]
        n = int(L * validation_split)
        n = min(L - 1, max(1, n))

        if shuffle:
            vi = np.random.choice(L, replace=False, size=(n,))
        else:
            vi = np.arange(n)

        ti = np.array(list(set(range(L)) - set(vi)))

        return (x[ti], y[ti]), (x[vi], y[vi])

    @staticmethod
    def _xy_batches(x, y, n_batch=16, batch_size=None, shuffle=True):
        """Make batches for an optional supervised dataset."""
        if x is None:
            return []

        if n_batch is None and batch_size is None:
            n_batch = 1

        return list(
            PhysicsGuidedNeuralNetwork.make_batches(
                x,
                y,
                n_batch=n_batch,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        )

    @staticmethod
    def _batch_or_none(batches, index):
        """Get a cycled batch or an empty dataset placeholder."""
        if not batches:
            return None, None

        return batches[index % len(batches)]

    def _get_state(self):
        """Get restorable trainable model state."""
        return {
            'layers': [layer.get_weights() for layer in self.layers],
            'baseline': self.baseline,
        }

    def _set_state(self, state):
        """Restore trainable model state."""
        for layer, weights in zip(self.layers, state['layers']):
            if weights:
                layer.set_weights(weights)

        if self._k0 is not None and state['baseline'] is not None:
            self._k0.assign(state['baseline'])

    def _get_multifidelity_grad(
        self,
        x_lf=None,
        y_lf=None,
        x_hf=None,
        y_hf=None,
        loss_weights=None,
        label_type='k',
        baseline_mu=None,
        baseline_stdev=None,
        baseline_epsilon=1e-6,
    ):
        """Get gradients for one LF/HF mini-batch."""
        with tf.GradientTape() as tape:
            out = self.calc_multifidelity_loss(
                x_lf=x_lf,
                y_lf=y_lf,
                x_hf=x_hf,
                y_hf=y_hf,
                loss_weights=loss_weights,
                label_type=label_type,
                baseline_mu=baseline_mu,
                baseline_stdev=baseline_stdev,
                baseline_epsilon=baseline_epsilon,
                training=True,
            )
            loss = out[0]
            weights = self.weights
            grad = tape.gradient(loss, weights)

        return grad, weights, out

    def run_multifidelity_gradient_descent(
        self,
        x_lf=None,
        y_lf=None,
        x_hf=None,
        y_hf=None,
        loss_weights=None,
        label_type='k',
        baseline_mu=None,
        baseline_stdev=None,
        baseline_epsilon=1e-6,
    ):
        """Run gradient descent for one LF/HF mini-batch."""
        grad, weights, out = self._get_multifidelity_grad(
            x_lf=x_lf,
            y_lf=y_lf,
            x_hf=x_hf,
            y_hf=y_hf,
            loss_weights=loss_weights,
            label_type=label_type,
            baseline_mu=baseline_mu,
            baseline_stdev=baseline_stdev,
            baseline_epsilon=baseline_epsilon,
        )
        grad_vars = [
            (g, w) for g, w in zip(grad, weights) if g is not None
        ]
        if not grad_vars:
            msg = 'No gradients were calculated for multi-fidelity training.'
            logger.error(msg)
            raise RuntimeError(msg)

        self._optimizer.apply_gradients(grad_vars)

        return out

    def _init_multifidelity_history(self):
        """Initialize or extend the training history columns."""
        columns = [
            'elapsed_time',
            'stage',
            'training_loss',
            'training_lf_loss',
            'training_hf_loss',
            'training_baseline_loss',
            'training_reg_loss',
            'validation_loss',
            'validation_lf_loss',
            'validation_hf_loss',
            'validation_baseline_loss',
            'validation_reg_loss',
            'learning_rate',
            'lambda_lf',
            'lambda_hf',
            'lambda_baseline',
        ]

        if self._history is None:
            self._history = pd.DataFrame(columns=columns)
            self._history.index.name = 'epoch'
        else:
            for column in columns:
                if column not in self._history:
                    self._history[column] = np.nan

    def fit_multifidelity(
        self,
        x_lf=None,
        y_lf=None,
        x_hf=None,
        y_hf=None,
        loss_weights=None,
        n_batch=16,
        batch_size=None,
        lf_batch_size=None,
        hf_batch_size=None,
        n_epoch=100,
        shuffle=True,
        validation_split=0.2,
        label_type='k',
        baseline_mu=None,
        baseline_stdev=None,
        baseline_epsilon=1e-6,
        learning_rate=None,
        early_stop=False,
        stop_kwargs=None,
        restore_best=True,
        stage=None,
        return_diagnostics=False,
    ):
        """Fit a residual PGNN with LF/HF supervised losses.

        This implements the document's shared-network loss:
        ``lambda_lf * L_LF + lambda_hf * L_HF + lambda_bn * L_BN``.
        The network output is Delta k. With ``label_type='k'``, labels are
        total k and predictions use ``k0 + Delta k``.

        Parameters
        ----------
        x_lf, y_lf : np.ndarray | pd.DataFrame | None
            Low-fidelity features and labels.
        x_hf, y_hf : np.ndarray | pd.DataFrame | None
            High-fidelity features and labels.
        loss_weights : dict | tuple | list | None
            LF/HF/baseline loss weights.
        n_batch : int | None
            Number of LF batches per epoch unless batch_size is supplied.
        batch_size : int | None
            Default batch size for both LF and HF data.
        lf_batch_size : int | None
            Optional LF-specific batch size.
        hf_batch_size : int | None
            Optional HF-specific batch size.
        n_epoch : int
            Maximum number of epochs.
        shuffle : bool
            Flag to shuffle train/validation split and batches.
        validation_split : float | dict
            Validation fraction. A dict may contain ``lf`` and ``hf`` keys.
        label_type : {'k', 'delta'}, optional
            Whether labels are total k or residual Delta k.
        baseline_mu : float | None, optional
            Baseline prior mean for this training run.
        baseline_stdev : float | None, optional
            Baseline prior standard deviation for this training run.
        baseline_epsilon : float, optional
            Small denominator stabilizer for the baseline prior.
        learning_rate : float | None, optional
            Optimizer learning rate override for this training run.
        early_stop : bool, optional
            Flag to stop after the monitored loss stops improving.
        stop_kwargs : dict | None, optional
            Early stop options. Defaults to
            ``{'monitor': 'validation_loss', 'patience': 50}``.
        restore_best : bool, optional
            Restore the best monitored state after early stopping.
        stage : str | None, optional
            Stage label stored in history.
        return_diagnostics : bool, optional
            Flag to return train/validation splits and history.

        Returns
        -------
        diagnostics : dict | None
            Training diagnostics if requested.
        """
        label_type = self._parse_label_type(label_type)
        if label_type == 'k' and self._k0 is None:
            msg = (
                'label_type="k" requires a residual baseline. Set '
                'baseline_init or use label_type="delta".'
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if learning_rate is not None:
            self.set_learning_rate(learning_rate)

        x_lf, y_lf, x_hf, y_hf = self.preflight_multifidelity_data(
            x_lf=x_lf, y_lf=y_lf, x_hf=x_hf, y_hf=y_hf
        )

        if isinstance(validation_split, dict):
            lf_validation_split = validation_split.get('lf', 0.0)
            hf_validation_split = validation_split.get('hf', 0.0)
        else:
            lf_validation_split = validation_split
            hf_validation_split = validation_split

        (x_lf, y_lf), (x_lf_val, y_lf_val) = self._train_val_split_xy(
            x_lf, y_lf, shuffle=shuffle, validation_split=lf_validation_split
        )
        (x_hf, y_hf), (x_hf_val, y_hf_val) = self._train_val_split_xy(
            x_hf, y_hf, shuffle=shuffle, validation_split=hf_validation_split
        )

        self._init_multifidelity_history()
        loss_weights = self._parse_multifidelity_loss_weights(loss_weights)
        stop_kwargs = stop_kwargs or {
            'monitor': 'validation_loss',
            'patience': 50,
            'min_delta': 0.0,
        }

        start_epoch = 0
        if len(self._history):
            start_epoch = int(max(self._history.index.values)) + 1

        t0 = time.time()
        best_value = np.inf
        best_state = None
        wait = 0

        for epoch in range(start_epoch, start_epoch + n_epoch):
            lf_bs = lf_batch_size if lf_batch_size is not None else batch_size
            hf_bs = hf_batch_size if hf_batch_size is not None else batch_size
            lf_batches = self._xy_batches(
                x_lf, y_lf, n_batch=n_batch, batch_size=lf_bs, shuffle=shuffle
            )
            hf_batches = self._xy_batches(
                x_hf, y_hf, n_batch=None, batch_size=hf_bs, shuffle=shuffle
            )

            n_steps = max(len(lf_batches), len(hf_batches))
            e_tr_loss = []
            e_tr_lf_loss = []
            e_tr_hf_loss = []
            e_tr_baseline_loss = []
            e_tr_reg_loss = []

            for b in range(n_steps):
                x_lf_batch, y_lf_batch = self._batch_or_none(lf_batches, b)
                x_hf_batch, y_hf_batch = self._batch_or_none(hf_batches, b)
                out = self.run_multifidelity_gradient_descent(
                    x_lf=x_lf_batch,
                    y_lf=y_lf_batch,
                    x_hf=x_hf_batch,
                    y_hf=y_hf_batch,
                    loss_weights=loss_weights,
                    label_type=label_type,
                    baseline_mu=baseline_mu,
                    baseline_stdev=baseline_stdev,
                    baseline_epsilon=baseline_epsilon,
                )
                tr_loss, tr_lf, tr_hf, tr_baseline, tr_reg = out
                e_tr_loss.append(tr_loss.numpy())
                e_tr_lf_loss.append(tr_lf.numpy())
                e_tr_hf_loss.append(tr_hf.numpy())
                e_tr_baseline_loss.append(tr_baseline.numpy())
                e_tr_reg_loss.append(tr_reg.numpy())
                logger.debug(
                    'Epoch {} batch {} multi-fidelity train loss: {:.2e} '
                    'for "{}"'.format(epoch, b, tr_loss, self.name)
                )

            has_val = x_lf_val is not None or x_hf_val is not None
            if has_val:
                val_out = self.calc_multifidelity_loss(
                    x_lf=x_lf_val,
                    y_lf=y_lf_val,
                    x_hf=x_hf_val,
                    y_hf=y_hf_val,
                    loss_weights=loss_weights,
                    label_type=label_type,
                    baseline_mu=baseline_mu,
                    baseline_stdev=baseline_stdev,
                    baseline_epsilon=baseline_epsilon,
                    training=False,
                )
                val_loss, val_lf, val_hf, val_baseline, val_reg = [
                    x.numpy() for x in val_out
                ]
            else:
                val_loss = np.nan
                val_lf = np.nan
                val_hf = np.nan
                val_baseline = np.nan
                val_reg = np.nan

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'stage'] = stage
            self._history.at[epoch, 'training_loss'] = np.mean(e_tr_loss)
            self._history.at[epoch, 'training_lf_loss'] = np.mean(
                e_tr_lf_loss
            )
            self._history.at[epoch, 'training_hf_loss'] = np.mean(
                e_tr_hf_loss
            )
            self._history.at[epoch, 'training_baseline_loss'] = np.mean(
                e_tr_baseline_loss
            )
            self._history.at[epoch, 'training_reg_loss'] = np.mean(
                e_tr_reg_loss
            )
            self._history.at[epoch, 'validation_loss'] = val_loss
            self._history.at[epoch, 'validation_lf_loss'] = val_lf
            self._history.at[epoch, 'validation_hf_loss'] = val_hf
            self._history.at[epoch, 'validation_baseline_loss'] = val_baseline
            self._history.at[epoch, 'validation_reg_loss'] = val_reg
            self._history.at[epoch, 'learning_rate'] = self._learning_rate
            self._history.at[epoch, 'lambda_lf'] = loss_weights['lf']
            self._history.at[epoch, 'lambda_hf'] = loss_weights['hf']
            self._history.at[epoch, 'lambda_baseline'] = (
                loss_weights['baseline']
            )

            logger.info(
                'Epoch {} train loss: {:.2e} val loss: {:.2e} for "{}"'
                .format(
                    epoch,
                    self._history.at[epoch, 'training_loss'],
                    self._history.at[epoch, 'validation_loss'],
                    self.name,
                )
            )

            if early_stop:
                monitor = stop_kwargs.get('monitor', 'validation_loss')
                patience = stop_kwargs.get('patience', 50)
                min_delta = stop_kwargs.get('min_delta', 0.0)
                if monitor not in self._history:
                    msg = 'Early stop monitor "{}" is not in history.'
                    logger.error(msg.format(monitor))
                    raise KeyError(msg.format(monitor))

                monitor_value = self._history.at[epoch, monitor]
                if pd.isna(monitor_value) and monitor.startswith(
                    'validation_'
                ):
                    train_monitor = monitor.replace(
                        'validation_', 'training_', 1
                    )
                    monitor_value = self._history.at[epoch, train_monitor]

                if monitor_value < best_value - min_delta:
                    best_value = monitor_value
                    best_state = self._get_state()
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    break

        if early_stop and restore_best and best_state is not None:
            self._set_state(best_state)

        diagnostics = {
            'x_lf': x_lf,
            'y_lf': y_lf,
            'x_hf': x_hf,
            'y_hf': y_hf,
            'x_lf_val': x_lf_val,
            'y_lf_val': y_lf_val,
            'x_hf_val': x_hf_val,
            'y_hf_val': y_hf_val,
            'history': self.history,
        }

        if return_diagnostics:
            return diagnostics
        return None

    def fit(
        self,
        x,
        y,
        p,
        n_batch=16,
        batch_size=None,
        n_epoch=10,
        shuffle=True,
        validation_split=0.2,
        p_kwargs=None,
        run_preflight=True,
        return_diagnostics=False,
    ):
        """Fit the neural network to data from x and y.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. Generally speaking, the data should
            always have the number of observations in the first axis and the
            number of features/channels in the last axis. Spatial and temporal
            dimensions can be used in intermediate axes.
        y : np.ndarray | pd.DataFrame
            Known output data in a >=2D array or DataFrame.
            Same dimension rules as x.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in >=2D
            array or DataFrame. Same dimension rules as x.
        n_batch : int | None
            Number of times to update the NN weights per epoch (number of
            mini-batches). The training data will be split into this many
            mini-batches and the NN will train on each mini-batch, update
            weights, then move onto the next mini-batch.
        batch_size : int | None
            Number of training samples per batch. This input is redundant to
            n_batch and will not be used if n_batch is not None.
        n_epoch : int
            Number of times to iterate on the training data.
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from x, y, and p.
        validation_split : float
            Fraction of x, y, and p to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.

        Returns
        -------
        diagnostics : dict
            Namespace of training parameters that can be used for diagnostics.
        """

        x, y, p = self.preflight_data(x, y, p)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=[
                    'elapsed_time',
                    'training_loss',
                    'training_nn_loss',
                    'training_p_loss',
                    'validation_loss',
                    'validation_nn_loss',
                    'validation_p_loss',
                ]
            )
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        val_splits = self.get_val_split(
            x, y, p, shuffle=shuffle, validation_split=validation_split
        )
        x, x_val = val_splits[0]
        y, y_val = val_splits[1]
        p, p_val = val_splits[2]

        if self._loss_weights[1] > 0 and run_preflight:
            self.preflight_p_fun(x_val, y_val, p_val, p_kwargs)

        t0 = time.time()
        for epoch in epochs:
            t_batch_iter = self.make_batches(
                x,
                y,
                p,
                n_batch=n_batch,
                batch_size=batch_size,
                shuffle=shuffle,
            )

            v_batch_iter = self.make_batches(
                x_val,
                y_val,
                p_val,
                n_batch=n_batch,
                batch_size=batch_size,
                shuffle=False,
            )

            e_tr_loss = []
            e_tr_nn_loss = []
            e_tr_p_loss = []

            e_val_loss = []
            e_val_nn_loss = []
            e_val_p_loss = []

            for b, (x_batch, y_batch, p_batch) in enumerate(t_batch_iter):
                b_out = self.run_gradient_descent(
                    x_batch, y_batch, p_batch, p_kwargs
                )
                b_tr_loss, b_tr_nn_loss, b_tr_p_loss = b_out
                e_tr_loss.append(b_tr_loss.numpy())
                e_tr_nn_loss.append(b_tr_nn_loss.numpy())
                e_tr_p_loss.append(b_tr_p_loss.numpy())
                logger.debug(
                    'Epoch {} batch {} train loss: {:.2e} for "{}"'.format(
                        epoch, b, b_tr_loss, self.name
                    )
                )

            for x_batch, y_batch, p_batch in v_batch_iter:
                y_val_pred = self.predict(x_batch, to_numpy=False)
                out = self.calc_loss(y_batch, y_val_pred, p_batch, p_kwargs)
                b_val_loss, b_val_nn_loss, b_val_p_loss = out
                e_val_loss.append(b_val_loss.numpy())
                e_val_nn_loss.append(b_val_nn_loss.numpy())
                e_val_p_loss.append(b_val_p_loss.numpy())

            e_tr_loss = np.mean(e_tr_loss)
            e_tr_nn_loss = np.mean(e_tr_nn_loss)
            e_tr_p_loss = np.mean(e_tr_p_loss)
            e_val_loss = np.mean(e_val_loss)
            e_val_nn_loss = np.mean(e_val_nn_loss)
            e_val_p_loss = np.mean(e_val_p_loss)

            logger.info(
                'Epoch {} train loss: {:.2e} val loss: {:.2e} for "{}"'.format(
                    epoch, e_tr_loss, e_val_loss, self.name
                )
            )

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss'] = e_tr_loss
            self._history.at[epoch, 'training_nn_loss'] = e_tr_nn_loss
            self._history.at[epoch, 'training_p_loss'] = e_tr_p_loss
            self._history.at[epoch, 'validation_loss'] = e_val_loss
            self._history.at[epoch, 'validation_nn_loss'] = e_val_nn_loss
            self._history.at[epoch, 'validation_p_loss'] = e_val_p_loss

        diagnostics = {
            'x': x,
            'y': y,
            'p': p,
            'x_val': x_val,
            'y_val': y_val,
            'p_val': p_val,
            'history': self.history,
        }

        if return_diagnostics:
            return diagnostics
        return None
