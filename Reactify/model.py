import tensorflow.keras as tfk


def model(
    length,
    n_filters_1=8,
    n_filters_2=16,
    filter_length_1=64,
    filter_length_2=8,
    dropout_1=0.1,
    dropout_2=0.1,
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
        optimizer="adam", loss="binary_crossentropy", metrics=["mean_absolute_error"]
    )

    return mdl
