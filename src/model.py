"""ATCNet architecture (parametrized -- config.MODEL_CONFIGS switches between full and mini).

Reimplemented directly from the official `models.py` / `attention_models.py`
(Altaheri/EEG-ATCNet: https://github.com/Altaheri/EEG-ATCNet, IEEE TII 2023,
10.1109/TII.2022.3197419), keeping the same three-block structure for both configurations
(Convolutional block -> sliding-window Attention block -> Temporal Convolutional block, fused by
averaging each window's per-class logits). Only the widths differ between 'ATCNet_full' and
'Mini-ATCNet' -- see README for the full per-hyperparameter comparison table and rationale.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import L2


def atcnet_conv_block(input_layer, F1, D, kern_length, pool_size, dropout, n_chans):
    """Convolutional (CV) block: temporal conv -> depthwise spatial conv -> separable conv.
    Mirrors ATCNet's Conv_block_ (models.py)."""
    F2 = F1 * D
    x = layers.Conv2D(F1, (kern_length, 1), padding='same', use_bias=False,
                       kernel_regularizer=L2(0.009), kernel_constraint=max_norm(0.6, axis=[0, 1, 2]))(input_layer)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.DepthwiseConv2D((1, n_chans), depth_multiplier=D, use_bias=False,
                                depthwise_regularizer=L2(0.009),
                                depthwise_constraint=max_norm(0.6, axis=[0, 1, 2]))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((8, 1))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(F2, (16, 1), padding='same', use_bias=False,
                       kernel_regularizer=L2(0.009), kernel_constraint=max_norm(0.6, axis=[0, 1, 2]))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((pool_size, 1))(x)
    x = layers.Dropout(dropout)(x)
    return x


def atcnet_mha_block(x, key_dim, num_heads, dropout=0.5):
    """Multi-head self-attention block: pre-LN -> MHA -> dropout -> residual add.
    Mirrors ATCNet's mha_block (attention_models.py)."""
    norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(norm, norm)
    attn = layers.Dropout(0.3)(attn)
    return layers.Add()([x, attn])


def atcnet_tcn_block(input_layer, input_dim, depth, kernel_size, filters, dropout, activation='elu'):
    """Temporal Convolutional Network block: dilated causal conv residual blocks.
    Mirrors ATCNet's TCN_block_ (models.py)."""
    def causal_conv(x, dilation_rate):
        x = layers.Conv1D(filters, kernel_size=kernel_size, dilation_rate=dilation_rate, activation='linear',
                           padding='causal', kernel_initializer='he_uniform',
                           kernel_regularizer=L2(0.009), kernel_constraint=max_norm(0.6, axis=[0, 1]))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return layers.Dropout(dropout)(x)

    block = causal_conv(input_layer, dilation_rate=1)
    block = causal_conv(block, dilation_rate=1)
    if input_dim != filters:
        residual = layers.Conv1D(filters, kernel_size=1, padding='same')(input_layer)
    else:
        residual = input_layer
    out = layers.Activation(activation)(layers.Add()([block, residual]))

    for i in range(depth - 1):
        dilation = 2 ** (i + 1)
        block = causal_conv(out, dilation_rate=dilation)
        block = causal_conv(block, dilation_rate=dilation)
        out = layers.Activation(activation)(layers.Add()([block, out]))
    return out


def build_atcnet(n_classes, n_chans, n_samples,
                  n_windows, eegn_F1, eegn_D, eegn_kernel_size, eegn_pool_size, eegn_dropout,
                  tcn_depth, tcn_kernel_size, tcn_filters, tcn_dropout,
                  mha_key_dim, mha_num_heads, model_name='ATCNet'):
    F2 = eegn_F1 * eegn_D

    inp = layers.Input(shape=(n_chans, n_samples, 1))
    x = layers.Permute((2, 1, 3))(inp)  # -> (time, channels, 1), channels-last conv over time

    conv_out = atcnet_conv_block(x, F1=eegn_F1, D=eegn_D, kern_length=eegn_kernel_size,
                                  pool_size=eegn_pool_size, dropout=eegn_dropout, n_chans=n_chans)
    conv_out = layers.Lambda(lambda t: t[:, :, -1, :])(conv_out)  # squeeze the spatial axis -> (time', F2)

    window_logits = []
    total_time = conv_out.shape[1]
    for i in range(n_windows):
        start = i
        end = total_time - n_windows + i + 1
        window = layers.Lambda(lambda t, s=start, e=end: t[:, s:e, :])(conv_out)

        window = atcnet_mha_block(window, key_dim=mha_key_dim, num_heads=mha_num_heads)
        window = atcnet_tcn_block(window, input_dim=F2, depth=tcn_depth, kernel_size=tcn_kernel_size,
                                   filters=tcn_filters, dropout=tcn_dropout)
        window_last = layers.Lambda(lambda t: t[:, -1, :])(window)  # last time step's feature vector
        window_logits.append(layers.Dense(n_classes, kernel_regularizer=L2(0.5))(window_last))

    fused = layers.Average()(window_logits) if len(window_logits) > 1 else window_logits[0]
    out = layers.Activation('softmax', name='softmax')(fused)

    return Model(inputs=inp, outputs=out, name=model_name)


class LRWarmup(tf.keras.callbacks.Callback):
    """Linearly ramp the learning rate from 0 up to `target_lr` over the first `warmup_epochs`
    epochs, then leave it alone (ReduceLROnPlateau takes over from there). Motivated by every fold
    in this project's pre-warmup run training the full epoch budget without early stopping ever
    triggering -- a cold start straight at the target LR into a heavily max_norm/L2-regularized
    architecture may not be using the epoch budget as effectively as it could.

    Note: assigns via `optimizer.learning_rate.assign(lr)` (Keras 3 API) rather than the Keras
    2 / TF1-era `tf.keras.backend.set_value(...)`, which raises
    `AttributeError: 'str' object has no attribute 'name'` under Keras 3 on current Kaggle images.
    """

    def __init__(self, target_lr, warmup_epochs):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate.assign(lr)
