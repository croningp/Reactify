import tensorflow.keras as tfk
import numpy as np


def model(
    length,
    n_filters_1=8,
    n_filters_2=16,
    n_filters_3=8,
    filter_shape_1=64,
    filter_shape_2=8,
    filter_shape_3=(8, 4),
    dropout_1=0.3,
    dropout_2=0.3,
    continuous=True,
    loss: str = None,
    dropout_on_inference=False,
):
    """Reactivity detection model.
    The returned model takes an input of the shape `(n_experiments, 2, len_spectrum)` where
    `input[i, 0, :]` is the normalized real part sum of the reactant spectra and
    `input[i, 1, :]` is the normalized real part of the reaction spectrum.
    The model returns a `(n_experiments, 1)` tensor if `continuous` is set to `True`,
    corresponding to a continuous reactivity scale between 0 and 1.
    Otherwise a `(n_experiments, 4)` tensor is returned, corresponding to 4 different
    reactivity labels from not reactive at all to very reactive.

    Args:
        n_filters_1 (int): Number of filters in convolutional layer 1.
        n_filters_2 (int): Number of filters in convolutional layer 2.
        n_filters_3 (int): Number of filters in convolutional layer 3.
        filter_shape_1 (int or Tuple[int]): Shape of filters in convolutional layer 1.
        filter_shape_2 (int or Tuple[int]): Shape of filters in convolutional layer 2.
        filter_shape_3 (int or Tuple[int]): Shape of filters in convolutional layer 3.
        dropout_1 (float): Dropout probability before first dense layer.
        dropout_2 (float): Dropout probability before second dense layer.
        continuous (bool): Whether the output is one continuous variable (True) or
            four corresponding to different reactivity classes.
        loss (str): 
        dropout_on_inference=False,
    """
    input = tfk.Input(shape=(2, length))
    dropout_kwarg = {"training": True} if dropout_on_inference else {}
    input1 = input[:, 0, :][:, :, np.newaxis]  # sample, NMR data point, channel
    input2 = input[:, 1, :][:, :, np.newaxis]

    # identical processing of each spectrum
    conv1 = tfk.layers.Conv1D(
        filters=n_filters_1, kernel_size=filter_shape_1, input_shape=(length,)
    )
    pool1 = tfk.layers.MaxPooling1D(pool_size=4)
    conv2 = tfk.layers.Conv1D(filters=n_filters_2, kernel_size=filter_shape_2)
    pool2 = tfk.layers.MaxPooling1D(pool_size=4)

    sig1 = pool2(conv2(pool1(conv1(input1))))
    sig2 = pool2(conv2(pool1(conv1(input2))))
    combined = tfk.layers.concatenate(
        [sig1[:, np.newaxis, ...], sig2[:, np.newaxis, ...]], axis=1
    )

    result = tfk.layers.Conv2D(
        n_filters_3, filter_shape_3, data_format="channels_first", padding="same"
    )(combined)
    result = tfk.layers.MaxPool2D(pool_size=(8, 2), data_format="channels_first")(
        result
    )
    result = tfk.layers.Reshape((n_filters_3, -1))(result)
    result = tfk.layers.Dropout(dropout_1)(result, **dropout_kwarg)
    result = tfk.layers.Dense(8, activation="relu")(result)
    result = tfk.layers.Flatten()(result)
    result = tfk.layers.Dropout(dropout_2)(result, **dropout_kwarg)
    result = tfk.layers.Dense(8, activation="relu")(result)
    result = tfk.layers.Dense(4, activation="softmax" if continuous else "relu")(result)
    if continuous:
        result = tfk.layers.Dense(1)(result)

    # return conv1, conv2, pool1, pool2, input1, input2, input, comb, result
    mdl = tfk.Model(inputs=input, outputs=result)

    loss = loss or ("mean_squared_error" if continuous else "binary_crossentropy")

    mdl.compile(
        optimizer="adam",
        loss=loss,
        metrics=["mean_squared_error", "mean_absolute_error"],
    )

    return mdl
