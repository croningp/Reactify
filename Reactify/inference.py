import argparse
from functools import reduce
from operator import add
from typing import Iterable

import numpy as np
from tensorflow import keras as tfk

from Reactify.model import model
from Reactify.nmr import NMRSpectrum, NMRDataset
from Reactify.util import register_params, retrieve_args


def infer_reactivity(
    model: model,
    rxn_spec: NMRSpectrum,
    reactant_spectra: Iterable[NMRSpectrum],
    reactant_weights=None,
):
    """Infer the reactivity of reaction spectrum `rxn_spec` relative to those of the
    reagents in `reagent_spectra`.

    Args:
        model: Trained model to use for inference.
        rxn_spec: NMR spectrum of the reaction mixture.
        reactant_spectra: NMR spectra for the individual reactants/reagents
        reactant_weights: Reaction stoichiometry, i.e. the concentration (relative to
            the corresponding spectrum in `reagent_spectra` of each reactant)

    """
    l = len(rxn_spec)
    ls = [len(s) for s in reactant_spectra]
    reactant_weights = reactant_weights or [1.0] * len(reactant_spectra)

    # Make sure reaction spectrum and all reagent spectra are the same length
    assert min(ls) == max(ls) == len(rxn_spec)

    test_point = np.zeros((1, 2, l))
    test_point[0, 0, :] = rxn_spec.normalize().spectrum.real
    test_point[0, 1, :] = (
        reduce(add, [w * s for (w, s) in zip(reactant_weights, reactant_spectra)])
        .normalize()
        .spectrum.real
    )

    return model(test_point)[0, ...]


def main(model_path: str, rxn_path: str, *reactant_paths: str):
    """
    Args:
        model_path: Path where pre-trained Tensorflow model is stored.
        rxn_path: Path to NMR spectrum whose reactivity needs to be assessed.
        reactant_paths: Paths to NMR spectra for reactants involved in the reaction.

    Returns:
        object:
    """
    model = tfk.models.load_model(model_path)
    rxn_spec = NMRSpectrum(rxn_path)
    reactant_dataset = NMRDataset(reactant_paths, target_length=len(rxn_spec))
    return infer_reactivity(model, rxn_spec, reactant_dataset[:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    register_params(parser, main)
    args = parser.parse_args()
    print(main(**retrieve_args(args, main)))
