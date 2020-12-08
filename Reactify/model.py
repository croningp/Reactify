import tensorflow.keras as tfk
import numpy as np


def model(
    length,
    n_filters_1=8,
    n_filters_2=16,
    filter_length_1=64,
    filter_length_2=8,
    dropout_1=0.1,
    dropout_2=0.1,
    loss="mean_squared_error"
):
    mdl = tfk.models.Sequential(
        [
            tfk.layers.Conv1D(
                filters=n_filters_1,
                kernel_size=filter_length_1,
                input_shape=(2, length),
                data_format="channels_first",
            ),
            tfk.layers.Permute((2, 1)),
            tfk.layers.MaxPooling1D(pool_size=filter_length_1 // 4),
            tfk.layers.Conv1D(filters=n_filters_2, kernel_size=filter_length_2),
            tfk.layers.MaxPooling1D(pool_size=filter_length_2 // 2),
            tfk.layers.Dense(16, activation="relu",),
            tfk.layers.Dropout(dropout_1),
            tfk.layers.Dense(8, activation="relu",),
            tfk.layers.Dropout(dropout_2),
            tfk.layers.Permute((2, 1)),
            tfk.layers.Dense(4, activation="relu",),
            tfk.layers.Flatten(),
            tfk.layers.Dense(2, activation="relu",),
            tfk.layers.Dense(1, activation="sigmoid",),
        ]
    )

    mdl.compile(
        optimizer="adam", loss=loss, metrics=["mean_squared_error", "mean_absolute_error"]
    )

    return mdl

def model2(
    length,
    n_filters_1=8,
    n_filters_2=16,
    filter_length_1=64,
    filter_length_2=8,
    dropout_1=0.1,
    dropout_2=0.1,
    loss="binary_crossentropy"
):
    input = tfk.Input(shape=(2, length))
    input1 = input[:, 0, :][:, :, np.newaxis] # sample, NMR data point, channel
    input2 = input[:, 1, :][:, :, np.newaxis]

    #layers
    conv1 = tfk.layers.Conv1D(filters=n_filters_1, kernel_size=filter_length_1, input_shape=(length,))
    pool1 = tfk.layers.MaxPooling1D(pool_size=filter_length_1 // 4)
    conv2 = tfk.layers.Conv1D(filters=n_filters_2, kernel_size=filter_length_2)
    pool2 = tfk.layers.MaxPooling1D(pool_size=filter_length_2 // 2)
    conv3 = tfk.layers.Conv2D(8, (8, 4), data_format="channels_first")
    pool3 = tfk.layers.MaxPool2D(pool_size=(8, 4), data_format="channels_first")
    reshape = tfk.layers.Reshape((8, -1))
    dense1 = tfk.layers.Dense(8, activation="relu")
    permute = tfk.layers.Permute((2, 1))
    dense2 = tfk.layers.Dense(4, activation="relu")
    flatten = tfk.layers.Flatten()
    dense3 = tfk.layers.Dense(8)
    dense4 = tfk.layers.Dense(4)


    sig1 = pool2(conv2(pool1(conv1(input1))))
    sig2 = pool2(conv2(pool1(conv1(input2))))
    comb = tfk.layers.concatenate([sig1[:, np.newaxis, ...], sig2[:, np.newaxis, ...]], axis=1)
    result = dense4(dense3(flatten(dense2(permute(dense1(reshape(pool3(conv3(comb)))))))))
    
    # return conv1, conv2, pool1, pool2, input1, input2, input, comb, result
    mdl = tfk.Model(inputs=input, outputs=result)

    mdl.compile(
        optimizer="adam", loss=loss, metrics=["mean_squared_error", "mean_absolute_error"]
    )

    return mdl

def model3(
    length,
    n_filters_1=8,
    n_filters_2=16,
    filter_length_1=64,
    filter_length_2=8,
    dropout_1=0.1,
    dropout_2=0.1,
    loss="binary_crossentropy"
):
    input = tfk.Input(shape=(2, length))
    input1 = input[:, 0, :][:, :, np.newaxis] # sample, NMR data point, channel
    input2 = input[:, 1, :][:, :, np.newaxis]

    #layers
    conv1 = tfk.layers.Conv1D(filters=n_filters_1, kernel_size=filter_length_1, input_shape=(length,))
    pool1 = tfk.layers.MaxPooling1D(pool_size=filter_length_1 // 4)
    conv2 = tfk.layers.Conv1D(filters=n_filters_2, kernel_size=filter_length_2)
    pool2 = tfk.layers.MaxPooling1D(pool_size=filter_length_2 // 2)
    conv3 = tfk.layers.Conv2D(8, (8, 4), data_format="channels_first")
    pool3 = tfk.layers.MaxPool2D(pool_size=(8, 4), data_format="channels_first")
    reshape = tfk.layers.Reshape((8, -1))
    dense1 = tfk.layers.Dense(8, activation="relu")
    flatten = tfk.layers.Flatten()
    dense2 = tfk.layers.Dense(8, activation="relu")
    dense3 = tfk.layers.Dense(4)


    sig1 = pool2(conv2(pool1(conv1(input1))))
    sig2 = pool2(conv2(pool1(conv1(input2))))
    comb = tfk.layers.concatenate([sig1[:, np.newaxis, ...], sig2[:, np.newaxis, ...]], axis=1)
    result = dense3(dense2(flatten(dense1(reshape(pool3(conv3(comb)))))))
    
    # return conv1, conv2, pool1, pool2, input1, input2, input, comb, result
    mdl = tfk.Model(inputs=input, outputs=result)

    mdl.compile(
        optimizer="adam", loss=loss, metrics=["mean_squared_error", "mean_absolute_error"]
    )

    return mdl

def model4(
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