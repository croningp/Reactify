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
from os import path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from .model import model as nn_model
from .util import register_params, retrieve_kwargs

sns.set(
    context="talk",
    style="ticks",
    font="Arial",
    font_scale=1.0,
    rc={"svg.fonttype": "none", "lines.linewidth": 1.5, "figure.autolayout": True,},
)


def generate_training_dataset(
    augmentation_factor=250,
    leave_out=100,
    max_shift=15,
    max_total_shift=450,
    coeff_wobble=0.1,
    circular_shift=False,
    quiet=False,
    one_hot=False,
):
    from .paper_dataset import (
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
    if one_hot:
        outcomes = np.zeros((n_examples, 4), dtype="float32")
    else:
        outcomes = np.zeros(n_examples, dtype="float32")

    val_inputs = np.zeros((leave_out, 2, min_length), dtype="float32")
    if one_hot:
        val_outcomes = np.zeros((leave_out, 4), dtype="float32")
    else:
        val_outcomes = np.zeros(leave_out, dtype="float32")

    cntr = 0

    # generate and augment training data points
    for i in range(augmentation_factor):
        if not quiet:
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
                if one_hot:
                    outcomes[cntr, label] = 1.0
                else:
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
                    .shift(shift, circular=circular_shift)
                    .spectrum.real
                )
                inputs[cntr, 1, :] = (
                    rxn_dataset[j].shift(shift, circular=circular_shift).spectrum.real
                )
                cntr += 1
    cntr = 0

    # generate validation data points
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
            if one_hot:
                val_outcomes[cntr, label] = 1.0
            else:
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
    epochs=15,
    batch_size=8,
    train=True,
    evaluate=True,
    plot=True,
    generation_kwargs={},
    model_kwargs={},
):
    print("Reading dataset ...")
    inputs, outcomes, val_inputs, val_outcomes = generate_training_dataset(
        **generation_kwargs
    )

    if train:
        print("Training model ...")
        model = nn_model(inputs.shape[-1], **model_kwargs)
        model.summary()

        try:
            model.fit(
                inputs,
                outcomes,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_inputs, val_outcomes),
            )
        except KeyboardInterrupt:
            pass

        model.save(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    if evaluate:
        print("Evaluating model ...")
        out = model(val_inputs)[:, 0]
        if plot:
            n_plots = len(out)
            f, axes = plt.subplots(nrows=n_plots, figsize=(10, 4 * n_plots))
            for inp, x, y, ax in zip(val_inputs, val_outcomes, out, axes):
                ax.plot(inp.T)
                ax.set_title(f"Expected {x:.2f}; found {y:.2}")
            f.savefig("result.svg")

        # plot confusion matrix
        out_classes = np.round(out * 3)
        expected_classes = np.round(val_outcomes * 3)
        cm = confusion_matrix(expected_classes, out_classes, labels=range(4))
        ax = sns.heatmap(cm, square=True, annot=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.figure.savefig(path.join(model_path, "confusion-matrix.svg"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    register_params(parser, main, exclude=["generation_kwargs", "model_kwargs"])
    register_params(parser, generate_training_dataset)
    register_params(parser, nn_model, exclude=["length"])
    args = parser.parse_args()
    main(
        **retrieve_kwargs(args, main),
        generation_kwargs=retrieve_kwargs(args, generate_training_dataset),
        model_kwargs=retrieve_kwargs(args, nn_model),
    )
