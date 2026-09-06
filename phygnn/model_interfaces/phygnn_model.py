# -*- coding: utf-8 -*-
# ruff: noqa: PLR0917
"""
TensorFlow Model
"""
import json
import logging
import os
import pprint

import numpy as np
import pandas as pd

from phygnn.bn_al2o3.constants import LOG_FEATURE_NAMES, RAW_FEATURE_NAMES
from phygnn.bn_al2o3.features import format_features
from phygnn.model_interfaces.base_model import ModelBase
from phygnn.phygnn import PhysicsGuidedNeuralNetwork
from phygnn.utilities.pre_processing import PreProcess

logger = logging.getLogger(__name__)


# 中文注释：这是 PGNN 的用户接口层，负责把表格数据、标准化、保存加载
# 和底层 PhysicsGuidedNeuralNetwork 训练逻辑连接起来。
class PhygnnModel(ModelBase):
    """
    Phygnn Model interface
    """

    # 中文注释：声明底层模型类，load() 会用它从 pkl 文件恢复神经网络。
    # Underlying model interface class. Used for loading models from disk
    MODEL_CLASS = PhysicsGuidedNeuralNetwork

    # 中文注释：文档中 BN/Al2O3 第一版方案固定的 6 个模型输入列。
    # 与 phygnn.bn_al2o3.constants 保持一致，避免重复声明。
    BN_AL2O3_FEATURE_NAMES = list(LOG_FEATURE_NAMES)

    # 中文注释：对数变换前的原始实验字段，供需要原始列的调用方使用。
    BN_AL2O3_RAW_FEATURE_NAMES = list(RAW_FEATURE_NAMES)

    # 中文注释：接口层默认把监督标签命名为总热导率 k。
    BN_AL2O3_LABEL_NAMES = ['k']

    # 中文注释：初始化接口对象，保存底层模型、特征名、标签名和预处理配置。
    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None, normalize=(True, False),
                 one_hot_categories=None):
        """
        Parameters
        ----------
        model : PhysicsGuidedNeuralNetwork
            PhysicsGuidedNeuralNetwork Model instance
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
        norm_params : dict, optional
            Dictionary mapping feature and label names (keys) to normalization
            parameters (mean, stdev), by default None
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
        """
        super().__init__(model, feature_names=feature_names,
                         label_names=label_names, norm_params=norm_params,
                         normalize=normalize,
                         one_hot_categories=one_hot_categories)

    # 中文注释：透出底层网络的层列表，便于检查实际网络结构。
    @property
    def layers(self):
        """
        Model layers

        Returns
        -------
        list
        """
        return self.model.layers

    # 中文注释：透出所有可训练权重，梯度计算和调试时会用到。
    @property
    def weights(self):
        """
        Get a list of layer weights for gradient calculations.

        Returns
        -------
        list
        """
        return self.model.weights

    # 中文注释：只获取 Dense 层的 kernel 权重，用于结构正则项。
    @property
    def kernel_weights(self):
        """
        Get a list of the NN kernel weights (tensors)

        (can be used for kernel regularization).

        Does not include input layer or dropout layers.
        Does include the output layer.

        Returns
        -------
        list
        """
        return self.model.kernel_weights

    # 中文注释：只获取 Dense 层的 bias 权重，用于 bias 正则项。
    @property
    def bias_weights(self):
        """
        Get a list of the NN bias weights (tensors)

        (can be used for bias regularization).

        Does not include input layer or dropout layers.
        Does include the output layer.

        Returns
        -------
        list
        """
        return self.model.bias_weights

    # 中文注释：当前 k0 基线值；文档里最终预测是 k0 + Delta_k。
    @property
    def baseline(self):
        """
        Current residual baseline k0 value.

        Returns
        -------
        float | None
        """
        return self.model.baseline

    # 中文注释：BN baseline 实验均值 mu_BN，用作 k0 的先验中心。
    @property
    def baseline_mu(self):
        """
        Experimental prior mean for k0.

        Returns
        -------
        float | None
        """
        return self.model.baseline_mu

    # 中文注释：BN baseline 实验标准差 sigma_BN，用作 k0 先验约束尺度。
    @property
    def baseline_stdev(self):
        """
        Experimental prior standard deviation for k0.

        Returns
        -------
        float | None
        """
        return self.model.baseline_stdev

    # 中文注释：标记 k0 是固定常数还是可学习的全局标量。
    @property
    def baseline_trainable(self):
        """
        Flag indicating whether k0 is trainable.

        Returns
        -------
        bool
        """
        return self.model.baseline_trainable

    # 中文注释：训练历史表，记录 loss、阶段、学习率和损失权重等信息。
    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self.model.history

    # 中文注释：记录模型创建时的重要包版本，便于后续复现实验。
    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return self.model.version_record

    # 中文注释：生成文档推荐的默认隐藏层，即 3 层、每层 32 个 Tanh 神经元。
    @classmethod
    def default_bn_al2o3_hidden_layers(cls, units=32, depth=3,
                                      activation='tanh'):
        """Get the document's first-pass 3x32 Tanh residual network."""
        return [{'units': units, 'activation': activation}
                for _ in range(depth)]

    # 中文注释：把实验表中的原始字段转换成神经网络实际使用的 6 维输入。
    @classmethod
    def format_bn_al2o3_features(cls, features, epsilon=1e-12):
        """Format raw BN/Al2O3 formulation fields into model inputs.

        Parameters
        ----------
        features : pandas.DataFrame | dict | np.ndarray
            Raw formulation table with ``D_s``, ``D_m``, ``D_l``, ``phi_s``,
            ``phi_m``, and ``E`` columns/keys, or an already formatted array.
        epsilon : float, optional
            Small positive value added before taking log(E + epsilon).

        Returns
        -------
        pandas.DataFrame | np.ndarray
            Formatted 6-column feature data:
            ``log_D_s, log_D_m, log_D_l, phi_s, phi_m, log_E``.
        """
        # 中文注释：委托给 phygnn.bn_al2o3.features，保证对数变换只有一处实现。
        return format_features(features, epsilon=epsilon)

    # 中文注释：更新底层模型的 k0 基线及其 BN 实验先验参数。
    def set_baseline(self, value, mu=None, stdev=None, trainable=None):
        """Set the residual baseline k0 and prior on the underlying model."""
        self.model.set_baseline(value, mu=mu, stdev=stdev,
                                trainable=trainable)

    # 中文注释：训练分阶段时可切换 Adam 学习率。
    def set_learning_rate(self, learning_rate):
        """Set the underlying optimizer learning rate."""
        self.model.set_learning_rate(learning_rate)

    # 中文注释：预测神经网络直接输出的残差 Delta_k，不加 BN 基线。
    def predict_delta(self, features, table=True, parse_kwargs=None,
                      predict_kwargs=None):
        """Predict residual Delta k from model inputs."""
        features = self._maybe_format_bn_al2o3_features(features)
        prediction = self.predict(features, table=table,
                                  parse_kwargs=parse_kwargs,
                                  predict_kwargs=predict_kwargs)
        if table and isinstance(prediction, pd.DataFrame):
            prediction.columns = ['Delta_k{}'.format(
                '' if len(prediction.columns) == 1 else '_{}'.format(i)
            ) for i in range(len(prediction.columns))]

        return prediction

    # 中文注释：预测最终热导率 k_pred，即底层残差输出加上 k0。
    def predict_k(self, features, table=True, parse_kwargs=None,
                  predict_kwargs=None):
        """Predict total k as k0 + Delta k."""
        if parse_kwargs is None:
            parse_kwargs = {}

        # 中文注释：BN/Al2O3 原始实验字段会在预测前自动转换成标准 6 列。
        features = self._maybe_format_bn_al2o3_features(features)

        # 中文注释：数组输入没有列名，因此需要根据维度补齐特征名。
        if isinstance(features, np.ndarray):
            n_features = features.shape[-1]
            if n_features == self.feature_dims:
                parse_kwargs.update({'names': self.feature_names})
            elif n_features == len(self.input_feature_names):
                parse_kwargs.update({'names': self.input_feature_names})
            else:
                msg = ('Number of features provided ({}) does not match '
                       'number of model features ({}) or number of input '
                       'features ({})'.format(n_features, self.feature_dims,
                                             len(self.input_feature_names)))
                logger.error(msg)
                raise RuntimeError(msg)

        features = self.parse_features(features, **parse_kwargs)
        if self.normalize_labels:
            msg = ('Residual total-k prediction expects native-label training '
                   'with normalize_labels=False.')
            logger.error(msg)
            raise RuntimeError(msg)

        if predict_kwargs is None:
            predict_kwargs = {}

        # 中文注释：底层模型负责真正计算 k0 + Delta_k。
        prediction = self.model.predict_k(features, **predict_kwargs)

        # 中文注释：默认返回 DataFrame，便于和实验表按列名拼接分析。
        if table and len(prediction.shape) in {1, 2}:
            columns = self.label_names or self.BN_AL2O3_LABEL_NAMES
            prediction = pd.DataFrame(prediction, columns=columns)

        return prediction

    # 中文注释：保留原项目的传统 PGNN 训练入口，兼容旧的 p_fun 物理损失用法。
    def train_model(self, features, labels, p,
                    n_batch=16, batch_size=None, n_epoch=10,
                    shuffle=True, validation_split=0.2, run_preflight=True,
                    return_diagnostics=False, p_kwargs=None,
                    parse_kwargs=None):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. A 2D input should have the shape:
            (n_observations, n_features). A 3D input should have the shape:
            (n_observations, n_timesteps, n_features). 4D inputs have not been
            tested and should be used with caution.
        labels : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame.
            Same dimension rules as features.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. Same dimension rules as features.
        n_batch : int
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
            from features, labels, and p.
        validation_split : float
            Fraction of features and labels to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function p_fun.
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.
        parse_kwargs : dict
            kwargs for cls.parse_features
        norm_labels : bool, optional
            Flag to normalize label, by default True

        Returns
        -------
        diagnostics : dict, optional
            Namespace of training parameters that can be used for diagnostics.
        """
        if parse_kwargs is None:
            parse_kwargs = {}

        # 中文注释：numpy 输入缺少列名时，按模型已有 feature_names 解释。
        if (isinstance(features, np.ndarray)
                and features.shape[-1] == self.feature_dims):
            parse_kwargs['names'] = self.feature_names

        # 中文注释：numpy 标签输入缺少列名时，按模型已有 label_names 解释。
        label_names = None
        if (isinstance(labels, np.ndarray)
                and labels.shape[-1] == self.label_dims):
            label_names = self.label_names

        # 中文注释：接口层先完成特征/标签解析和标准化，再交给底层训练循环。
        x = self.parse_features(features, **parse_kwargs)
        y = self.parse_labels(labels, names=label_names)

        diagnostics = self.model.fit(x, y, p,
                                     n_batch=n_batch,
                                     batch_size=batch_size,
                                     n_epoch=n_epoch,
                                     shuffle=shuffle,
                                     validation_split=validation_split,
                                     p_kwargs=p_kwargs,
                                     run_preflight=run_preflight,
                                     return_diagnostics=return_diagnostics)

        return diagnostics

    # 中文注释：仅当模型使用 BN/Al2O3 标准 6 特征时，自动转换原始实验字段。
    def _maybe_format_bn_al2o3_features(self, features):
        """Format BN/Al2O3 raw fields when this model uses that schema."""
        if (features is not None
                and self.feature_names == self.BN_AL2O3_FEATURE_NAMES
                and not isinstance(features, np.ndarray)):
            features = self.format_bn_al2o3_features(features)

        return features

    # 中文注释：多保真训练允许 LF 或 HF 某一组为空，这里统一做可选数据解析。
    def _parse_optional_training_dataset(self, features, labels,
                                         parse_kwargs=None):
        """Parse an optional supervised dataset for multi-fidelity training."""
        if features is None and labels is None:
            return None, None

        # 中文注释：防止只传特征或只传标签造成训练集错位。
        if features is None or labels is None:
            msg = 'features and labels must both be provided or both be None.'
            logger.error(msg)
            raise ValueError(msg)

        parse_kwargs = dict(parse_kwargs or {})
        features = self._maybe_format_bn_al2o3_features(features)

        # 中文注释：数组输入需要显式绑定列名，DataFrame 输入可从列名推断。
        if (isinstance(features, np.ndarray)
                and features.shape[-1] == self.feature_dims):
            parse_kwargs['names'] = self.feature_names

        label_names = None
        if (isinstance(labels, np.ndarray)
                and labels.shape[-1] == self.label_dims):
            label_names = self.label_names

        x = self.parse_features(features, **parse_kwargs)
        y = self.parse_labels(labels, names=label_names)

        return x, y

    # 中文注释：文档方案的单阶段多保真训练接口，LF/HF 共用同一个网络。
    def train_multifidelity(self, lf_features=None, lf_labels=None,
                            hf_features=None, hf_labels=None,
                            loss_weights=None,
                            n_batch=16, batch_size=None,
                            lf_batch_size=None, hf_batch_size=None,
                            n_epoch=100, shuffle=True,
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
                            parse_kwargs=None):
        """Train with LF/HF losses using the residual PGNN formulation.

        LF and HF labels are total k by default. The underlying network output
        remains Delta k, and ``predict_k`` returns ``k0 + Delta k``.
        """
        # 中文注释：当标签是总热导率 k 时，不能再对标签做标准化；
        # 否则 k0 相加会失去物理尺度。
        if label_type == 'k' and self.normalize_labels:
            msg = ('Multi-fidelity total-k training expects '
                   'normalize_labels=False.')
            logger.error(msg)
            raise RuntimeError(msg)

        # 中文注释：分别解析 LF 和 HF 数据，解析后都使用相同的特征标准化参数。
        x_lf, y_lf = self._parse_optional_training_dataset(
            lf_features, lf_labels, parse_kwargs=parse_kwargs
        )
        x_hf, y_hf = self._parse_optional_training_dataset(
            hf_features, hf_labels, parse_kwargs=parse_kwargs
        )

        # 中文注释：底层 PhysicsGuidedNeuralNetwork 负责实际 loss、
        # 梯度和 Adam 更新。
        diagnostics = self.model.fit_multifidelity(
            x_lf=x_lf,
            y_lf=y_lf,
            x_hf=x_hf,
            y_hf=y_hf,
            loss_weights=loss_weights,
            n_batch=n_batch,
            batch_size=batch_size,
            lf_batch_size=lf_batch_size,
            hf_batch_size=hf_batch_size,
            n_epoch=n_epoch,
            shuffle=shuffle,
            validation_split=validation_split,
            label_type=label_type,
            baseline_mu=baseline_mu,
            baseline_stdev=baseline_stdev,
            baseline_epsilon=baseline_epsilon,
            learning_rate=learning_rate,
            early_stop=early_stop,
            stop_kwargs=stop_kwargs,
            restore_best=restore_best,
            stage=stage,
            return_diagnostics=return_diagnostics,
        )

        return diagnostics

    # 中文注释：文档推荐的两阶段流程：先 LF 预训练，再 LF+HF 高权重校正。
    def train_two_stage_multifidelity(self, lf_features, lf_labels,
                                      hf_features, hf_labels,
                                      stage1_kwargs=None,
                                      stage2_kwargs=None,
                                      parse_kwargs=None,
                                      return_diagnostics=False):
        """Run LF pretraining followed by LF+HF multi-fidelity correction."""
        lambda_bn = self.model.baseline_loss_weight
        # 中文注释：Stage 1 只用 LF 学整体趋势，并用 BN baseline 先验约束 k0。
        stage1_defaults = {
            'loss_weights': {'lf': 1.0, 'hf': 0.0, 'baseline': lambda_bn},
            'n_batch': None,
            'batch_size': 256,
            'n_epoch': 500,
            'validation_split': {'lf': 0.2, 'hf': 0.0},
            'learning_rate': 1e-3,
            'early_stop': True,
            'stop_kwargs': {'monitor': 'validation_loss', 'patience': 50},
            'stage': 'lf_pretrain',
        }
        # 中文注释：Stage 2 同时使用 LF 和 HF，HF 默认给更高权重校正 LF 偏差。
        stage2_defaults = {
            'loss_weights': {'lf': 1.0, 'hf': 10.0, 'baseline': lambda_bn},
            'n_batch': None,
            'batch_size': 256,
            'n_epoch': 1000,
            'validation_split': {'lf': 0.2, 'hf': 0.0},
            'learning_rate': 1e-4,
            'early_stop': True,
            'stop_kwargs': {'monitor': 'validation_loss', 'patience': 50},
            'stage': 'multifidelity_correction',
        }

        # 中文注释：允许调用方覆盖 epoch、batch、loss 权重、
        # early stopping 等默认设置。
        if stage1_kwargs is not None:
            stage1_defaults.update(stage1_kwargs)
        if stage2_kwargs is not None:
            stage2_defaults.update(stage2_kwargs)

        # 中文注释：第一阶段只传 LF 数据，对应文档里的低保真预训练。
        diag1 = self.train_multifidelity(
            lf_features=lf_features,
            lf_labels=lf_labels,
            parse_kwargs=parse_kwargs,
            return_diagnostics=True,
            **stage1_defaults,
        )
        # 中文注释：第二阶段传 LF + 当前 HF 训练折，对应多保真校正。
        diag2 = self.train_multifidelity(
            lf_features=lf_features,
            lf_labels=lf_labels,
            hf_features=hf_features,
            hf_labels=hf_labels,
            parse_kwargs=parse_kwargs,
            return_diagnostics=True,
            **stage2_defaults,
        )

        diagnostics = {
            'stage1': diag1,
            'stage2': diag2,
            'history': self.history,
        }

        if return_diagnostics:
            return diagnostics
        return None

    # 中文注释：保存接口层配置到 json，同时保存底层 PGNN 到 pkl。
    def save_model(self, path):
        """
        Save phygnn model to path.

        Parameters
        ----------
        path : str
            Target model save path. Can be a target .json, .pkl, or a directory
            that will be created+populated with a json parameters file and a
            `.pkl` file for the underlying PhysicsGuidedNeuralNetwork.
        """

        # 中文注释：支持传入目录、json 路径或 pkl 路径，内部统一成 json+pkl。
        json_path = os.path.abspath(path)
        if json_path.endswith(('.json', '.pkl')):
            dir_path = os.path.dirname(json_path)
            if json_path.endswith('.pkl'):
                json_path = json_path.replace('.pkl', '.json')
        else:
            dir_path = json_path
            fn = os.path.basename(json_path) + '.json'
            json_path = os.path.join(dir_path, fn)

        pkl_path = json_path.replace('.json', '.pkl')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 中文注释：这里只保存接口层元数据，真正神经网络权重
        # 在下面的 pkl 中保存。
        model_params = {'feature_names': self.feature_names,
                        'label_names': self.label_names,
                        'norm_params': self.normalization_parameters,
                        'normalize': (self.normalize_features,
                                      self.normalize_labels),
                        'version_record': self.version_record,
                        'one_hot_categories': self.one_hot_categories,
                        }

        model_params = self.dict_json_convert(model_params)
        with open(json_path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

        self.model.save(pkl_path)

    # 中文注释：兼容旧接口，用于调整传统 PGNN 的普通损失和物理损失权重。
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
        self.model._loss_weights = loss_weights

    # 中文注释：通用 PGNN 构建入口，适合已有 p_fun 的原始 phygnn 用法。
    @classmethod
    def build(cls, p_fun, feature_names, label_names,
              normalize=(True, False),
              one_hot_categories=None,
              loss_weights=(0.5, 0.5),
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
              name=None):
        """
        Build phygnn model from given features, layers and kwargs

        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
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
            Initial value for residual baseline k0.
        baseline_mu : float | None, optional
            Experimental prior mean for k0. Defaults to baseline_init.
        baseline_stdev : float | None, optional
            Experimental prior standard deviation for k0.
        baseline_trainable : bool, optional
            Flag to make k0 a trainable scalar.
        baseline_loss_weight : float, optional
            Default k0 prior loss weight for multi-fidelity training.
        name : None | str
            Optional model name for debugging.

        Returns
        -------
        model : PhygnnModel
            Initialized PhygnnModel instance
        """
        if isinstance(label_names, str):
            label_names = [label_names]

        # 中文注释：如果有分类特征，这里先展开成 one-hot 后的真实输入列。
        if one_hot_categories is not None:
            check_names = feature_names + label_names
            PreProcess.check_one_hot_categories(one_hot_categories,
                                                feature_names=check_names)
            feature_names = cls.make_one_hot_feature_names(feature_names,
                                                           one_hot_categories)

        n_features = None if feature_names is None else len(feature_names)
        n_labels = None if label_names is None else len(label_names)

        # 中文注释：创建底层 PhysicsGuidedNeuralNetwork，
        # 所有训练细节都由它处理。
        model = PhysicsGuidedNeuralNetwork(p_fun,
                                           loss_weights=loss_weights,
                                           n_features=n_features,
                                           n_labels=n_labels,
                                           hidden_layers=hidden_layers,
                                           input_layer=input_layer,
                                           output_layer=output_layer,
                                           layers_obj=layers_obj,
                                           metric=metric,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           history=history,
                                           kernel_reg_rate=kernel_reg_rate,
                                           kernel_reg_power=kernel_reg_power,
                                           bias_reg_rate=bias_reg_rate,
                                           bias_reg_power=bias_reg_power,
                                           baseline_init=baseline_init,
                                           baseline_mu=baseline_mu,
                                           baseline_stdev=baseline_stdev,
                                           baseline_trainable=(
                                               baseline_trainable),
                                           baseline_loss_weight=(
                                               baseline_loss_weight),
                                           feature_names=feature_names,
                                           output_names=label_names,
                                           name=name)

        # 中文注释：用接口层包住底层模型，获得 DataFrame、
        # 标准化和保存加载能力。
        model = cls(model, feature_names=feature_names,
                    label_names=label_names, normalize=normalize,
                    one_hot_categories=one_hot_categories)

        return model

    # 中文注释：文档方案专用构建入口，默认创建 BN/Al2O3 残差多保真 PGNN。
    @classmethod
    def build_bn_al2o3(cls, baseline_mu, baseline_stdev,
                       baseline_trainable=True,
                       baseline_loss_weight=1.0,
                       feature_names=None,
                       label_names=None,
                       hidden_layers=None,
                       metric='mse',
                       learning_rate=1e-3,
                       loss_weights=(1.0, 0.0),
                       normalize=(True, False),
                       **kwargs):
        """Build the document's first-pass BN/Al2O3 residual PGNN.

        The default architecture is ``6 -> 32 -> 32 -> 32 -> 1`` with Tanh
        hidden activations and a linear residual output. The baseline k0 is
        initialized from ``baseline_mu`` and can be weakly adjusted during
        multi-fidelity training when ``baseline_trainable=True``.
        """
        # 中文注释：默认使用文档定义的 6 个输入和总热导率 k 标签。
        if feature_names is None:
            feature_names = cls.BN_AL2O3_FEATURE_NAMES
        if label_names is None:
            label_names = cls.BN_AL2O3_LABEL_NAMES
        # 中文注释：默认网络结构是 6 -> 32 -> 32 -> 32 -> 1，隐藏层为 Tanh。
        if hidden_layers is None:
            hidden_layers = cls.default_bn_al2o3_hidden_layers()

        # 中文注释：k0 初始化为 mu_BN，并将 mu_BN/sigma_BN 传入底层先验损失。
        return cls.build(
            None,
            feature_names,
            label_names,
            normalize=normalize,
            loss_weights=loss_weights,
            hidden_layers=hidden_layers,
            output_layer={'units': len(label_names)},
            metric=metric,
            learning_rate=learning_rate,
            baseline_init=baseline_mu,
            baseline_mu=baseline_mu,
            baseline_stdev=baseline_stdev,
            baseline_trainable=baseline_trainable,
            baseline_loss_weight=baseline_loss_weight,
            **kwargs,
        )

    # 中文注释：兼容旧接口，一步完成普通 PGNN 的构建、训练和可选保存。
    @classmethod
    def build_trained(cls, p_fun, features, labels, p,
                      normalize=(True, False),
                      one_hot_categories=None,
                      loss_weights=(0.5, 0.5),
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
                      n_batch=16,
                      batch_size=None,
                      n_epoch=10,
                      shuffle=True,
                      validation_split=0.2,
                      run_preflight=True,
                      return_diagnostics=False,
                      p_kwargs=None,
                      parse_kwargs=None,
                      save_path=None,
                      name=None):
        """
        Build phygnn model from given features, layers and
        kwargs and then train with given labels and kwargs

        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
        features : np.ndarray | pd.DataFrame
            Feature data in a >=2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm. A 2D input should have the shape:
            (n_observations, n_features). A 3D input should have the shape:
            (n_observations, n_timesteps, n_features). 4D inputs have not been
            tested and should be used with caution.
        labels : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame.
            Same dimension rules as features.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. Same dimension rules as features.
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
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
        output_layer : None } bool | list | dict
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
            Initial value for residual baseline k0.
        baseline_mu : float | None, optional
            Experimental prior mean for k0. Defaults to baseline_init.
        baseline_stdev : float | None, optional
            Experimental prior standard deviation for k0.
        baseline_trainable : bool, optional
            Flag to make k0 a trainable scalar.
        baseline_loss_weight : float, optional
            Default k0 prior loss weight for multi-fidelity training.
        n_batch : int
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
            from features and labels.
        validation_split : float
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.
            Fraction of features and labels to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function p_fun.
        parse_kwargs : dict
            kwargs for cls.parse_features
        norm_labels : bool, optional
            Flag to normalize label, by default True
        save_path : str, optional
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json, by default None
        name : None | str
            Optional model name for debugging.

        Returns
        -------
        model : TfModel
            Initialized and trained TfModel obj
        diagnostics : dict, optional
            Namespace of training parameters that can be used for diagnostics.
        """

        # 中文注释：从 DataFrame/dict/ndarray 推断特征名和标签名。
        _, feature_names = cls._parse_data_names(features, fallback_prefix='F')
        _, label_names = cls._parse_data_names(labels, fallback_prefix='L')

        # 中文注释：先构建模型，再调用传统 train_model() 训练。
        model = cls.build(p_fun, feature_names, label_names,
                          normalize=normalize,
                          one_hot_categories=one_hot_categories,
                          loss_weights=loss_weights,
                          hidden_layers=hidden_layers,
                          input_layer=input_layer,
                          output_layer=output_layer,
                          layers_obj=layers_obj,
                          metric=metric,
                          optimizer=optimizer,
                          learning_rate=learning_rate,
                          history=history,
                          kernel_reg_rate=kernel_reg_rate,
                          kernel_reg_power=kernel_reg_power,
                          bias_reg_rate=bias_reg_rate,
                          bias_reg_power=bias_reg_power,
                          baseline_init=baseline_init,
                          baseline_mu=baseline_mu,
                          baseline_stdev=baseline_stdev,
                          baseline_trainable=baseline_trainable,
                          baseline_loss_weight=baseline_loss_weight,
                          name=name)

        diagnostics = model.train_model(features, labels, p,
                                        n_batch=n_batch,
                                        batch_size=batch_size,
                                        n_epoch=n_epoch,
                                        shuffle=shuffle,
                                        validation_split=validation_split,
                                        run_preflight=run_preflight,
                                        return_diagnostics=return_diagnostics,
                                        p_kwargs=p_kwargs,
                                        parse_kwargs=parse_kwargs)

        # 中文注释：如果指定 save_path，训练结束后立刻保存当前模型。
        if save_path is not None:
            model.save_model(save_path)

        if diagnostics:
            return model, diagnostics
        return model

    # 中文注释：从磁盘恢复接口层 json 和底层 PGNN pkl。
    @classmethod
    def load(cls, path):
        """
        Load model from model path.

        Parameters
        ----------
        path : str
            Directory path for PhygnnModel to load model from. There should be
            a saved model directory with a json file for the PhygnnModel
            framework and a `.pkl` file for the underlying
            PhysicsGuidedNeuralNetwork.

        Returns
        -------
        model : PhygnnModel
            Loaded PhygnnModel from disk.
        """
        # 中文注释：支持传目录、json 或 pkl，内部统一定位到底层 pkl 文件。
        path = os.path.abspath(path)
        if not path.endswith(('.json', '.pkl')):
            pkl_path = os.path.join(path, os.path.basename(path) + '.pkl')
        elif path.endswith('.json'):
            pkl_path = path.replace('.json', '.pkl')
        elif path.endswith('.pkl'):
            pkl_path = path

        if not os.path.exists(pkl_path):
            e = ('{} does not exist'.format(pkl_path))
            logger.error(e)
            raise OSError(e)

        # 中文注释：先恢复底层神经网络及其权重、优化器、k0 等状态。
        loaded = cls.MODEL_CLASS.load(pkl_path)

        # 中文注释：再读取接口层元数据，例如特征名、标签名和标准化参数。
        json_path = pkl_path.replace('.pkl', '.json')
        if not os.path.exists(json_path):
            e = ('{} does not exist'.format(json_path))
            logger.error(e)
            raise OSError(e)

        with open(json_path) as f:
            model_params = json.load(f)

        if 'version_record' in model_params:
            version_record = model_params.pop('version_record')
            logger.info('Loading model from disk that was created with the '
                        'following package versions: \n{}'
                        .format(pprint.pformat(version_record, indent=4)))

        model = cls(loaded, **model_params)

        return model
