import tensorflow.keras as tfk
import numpy as np


def model(
    length,
    n_filters_1=8,
    n_filters_2=16,
    filter_length_1=64,
    filter_length_2=8,
    dropout_1=0.3,
    dropout_2=0.3,
    continuous=True,
    loss:str=None,
):
    input = tfk.Input(shape=(2, length))
    input1 = input[:, 0, :][:, :, np.newaxis] # sample, NMR data point, channel
    input2 = input[:, 1, :][:, :, np.newaxis]

    # identical processing of each spectrum
    conv1 = tfk.layers.Conv1D(filters=n_filters_1, kernel_size=filter_length_1, input_shape=(length,))
    pool1 = tfk.layers.MaxPooling1D(pool_size=4)
    conv2 = tfk.layers.Conv1D(filters=n_filters_2, kernel_size=filter_length_2)
    pool2 = tfk.layers.MaxPooling1D(pool_size=4)

    sig1 = pool2(conv2(pool1(conv1(input1))))
    sig2 = pool2(conv2(pool1(conv1(input2))))
    combined = tfk.layers.concatenate([sig1[:, np.newaxis, ...], sig2[:, np.newaxis, ...]], axis=1)

    result = tfk.layers.Conv2D(8, (8, 4), data_format="channels_first", padding="same")(combined)
    result = tfk.layers.MaxPool2D(pool_size=(8,2), data_format="channels_first")(result)
    result = tfk.layers.Reshape((8, -1))(result)
    result = tfk.layers.Dropout(dropout_1)(result)
    result = tfk.layers.Dense(8, activation="relu")(result)
    result = tfk.layers.Flatten()(result)
    result = tfk.layers.Dropout(dropout_2)(result)
    result = tfk.layers.Dense(8, activation="relu")(result)
    result = tfk.layers.Dense(4, activation="softmax" if continuous else "relu")(result)
    if continuous:
        result = tfk.layers.Dense(1)(result)
    
    # return conv1, conv2, pool1, pool2, input1, input2, input, comb, result
    mdl = tfk.Model(inputs=input, outputs=result)

    loss = loss or ("mean_squared_error" if continuous else "binary_crossentropy")

    mdl.compile(
        optimizer="adam", loss=loss, metrics=["mean_squared_error", "mean_absolute_error"]
    )

    return mdl