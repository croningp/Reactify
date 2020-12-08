import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import argparse
from functools import reduce
from operator import add

import numpy as np

from .model import model as nn_model
from .util import register_params, retrieve_params


def generate_training_set(
    augmentation_factor=100,
    leave_out=100,
    max_shift=15,
    max_total_shift=450,
    coeff_wobble=0.1,
):
    from .dataset import (
        FOLDER_PHOTO,
        FOLDER_SIMPLE2R,
        FOLDER_SIMPLE6R,
        df_photo,
        df_simple2r,
        df_simple6r,
        get_reagents,
        min_length,
        photo_dataset,
        photo_numbers,
        photo_reaction_dataset,
        simple2r_reaction_dataset,
        simple6r_reaction_dataset,
        simple_dataset,
        simple_numbers,
    )

    n_examples = augmentation_factor * (
        len(df_photo) + len(df_simple2r) + len(df_simple6r) - leave_out
    )

    inputs = np.zeros((n_examples, 2, min_length), dtype="float32")
    outcomes = np.zeros(n_examples, dtype="float32")

    val_inputs = np.zeros((leave_out, 2, min_length), dtype="float32")
    val_outcomes = np.zeros(leave_out, dtype="float32")

    cntr = 0

    for i in range(augmentation_factor):
        print(f"Augmentation @ {i+1}x")
        for folder, df, numbers, dataset, rxn_dataset in [
            (
                FOLDER_PHOTO,
                df_photo[:-leave_out],
                photo_numbers,
                photo_dataset,
                photo_reaction_dataset,
            ),
            (
                FOLDER_SIMPLE2R,
                df_simple2r,
                simple_numbers,
                simple_dataset,
                simple2r_reaction_dataset,
            ),
            (
                FOLDER_SIMPLE6R,
                df_simple6r,
                simple_numbers,
                simple_dataset,
                simple6r_reaction_dataset,
            ),
        ]:
            for j, (rxn, label) in df.reset_index(drop=True).iterrows():
                shift = np.random.randint(max_total_shift * 2) - max_total_shift
                outcomes[cntr] = label / 3
                inputs[cntr, 0, :] = (
                    reduce(
                        add,
                        [
                            dataset[numbers[r]].shift(
                                np.random.randint(max_shift * 2) - max_shift
                            )
                            * (np.random.randn() * coeff_wobble + 1.0)
                            * coeff
                            for r, coeff in get_reagents(folder, rxn)[0].items()
                            if r in numbers
                        ],
                    )
                    .normalize()
                    .shift(shift)
                    .spectrum.real
                )
                inputs[cntr, 1, :] = rxn_dataset[j].shift(shift).spectrum.real
                cntr += 1
    cntr = 0

    for folder, df, numbers, dataset, rxn_dataset in [
        (
            FOLDER_PHOTO,
            df_photo[-leave_out:],
            photo_numbers,
            photo_dataset,
            photo_reaction_dataset[-leave_out:],
        )
    ]:
        for j, (rxn, label) in df.reset_index(drop=True).iterrows():
            val_outcomes[cntr] = label / 3
            val_inputs[cntr, 0, :] = (
                reduce(
                    add,
                    [
                        dataset[numbers[r]]
                        for r, coeff in get_reagents(folder, rxn)[0].items()
                        if r in numbers
                    ],
                )
                .normalize()
                .spectrum.real
            )
            val_inputs[cntr, 1, :] = rxn_dataset[j].spectrum.real
            cntr += 1

    return inputs, outcomes, val_inputs, val_outcomes


def main(
    model_path: str,
    epochs=50,
    train=True,
    evaluate=True,
    generation_kwargs={},
    model_kwargs={},
):
    inputs, outcomes, val_inputs, val_outcomes = generate_training_set(
        **generation_kwargs
    )

    if train:
        model = nn_model(inputs.shape[-1], **model_kwargs)
        model.summary()

        model.fit(
            inputs, outcomes, epochs=epochs, validation_data=(val_inputs, val_outcomes),
        )

        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    if evaluate:
        import seaborn as sns
        from matplotlib import pyplot as plt

        sns.set(
            context="talk",
            style="ticks",
            font="Arial",
            font_scale=1.0,
            rc={
                "svg.fonttype": "none",
                "lines.linewidth": 1.5,
                "figure.autolayout": True,
            },
        )
        out = model(val_inputs)[:, 0]
        n_plots = len(out)
        f, axes = plt.subplots(nrows=n_plots, figsize=(10, 4 * n_plots))
        for inp, x, y, ax in zip(val_inputs, val_outcomes, out, axes):
            ax.plot(inp.T)
            ax.set_title(f"Expected {x:.2f}; found {y:.2}")
        f.savefig("result.svg")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    register_params(parser, main, exclude=["generation_kwargs", "model_kwargs"])
    register_params(parser, generate_training_set)
    register_params(parser, nn_model, exclude=["length"])
    args = parser.parse_args()

    main(
        **retrieve_params(args, main),
        generation_kwargs=retrieve_params(args, generate_training_set),
        model_kwargs=retrieve_params(args, nn_model),
    )
