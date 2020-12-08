import glob
from functools import lru_cache
from os import path

import pandas as pd

from .config import DATA_DIR
from .nmr import NMRDataset

FOLDER_PHOTO_REAGENTS = path.join(DATA_DIR, "002_photo_space/reagents")
FOLDER_SIMPLE_REAGENTS = path.join(DATA_DIR, "001_simple_space/reagents")
FOLDER_PHOTO = path.join(DATA_DIR, "20180418-1809-photochemical_space")
FOLDER_SIMPLE2R = path.join(DATA_DIR, "20171031-1642-Simple_space-DMSO")
FOLDER_SIMPLE6R = path.join(DATA_DIR, "20180404-1917-6R-Simple_space-DMSO")

df_photo = pd.read_csv(path.join(DATA_DIR, "photo.csv"))
df_simple2r = pd.read_csv(path.join(DATA_DIR, "simple_2r.csv"))
df_simple6r = pd.read_csv(path.join(DATA_DIR, "simple_6r.csv"))

paths_photo = {
    path.basename(d): d
    for d in glob.glob(path.join(FOLDER_PHOTO, "**", "[0-9]*-NMR-*[0-9]"))
}
paths_simple2r = {
    path.basename(d): d
    for d in glob.glob(path.join(FOLDER_SIMPLE2R, "**", "[0-9]*-NMR-*[0-9]"))
}
paths_simple6r = {
    path.basename(d): d
    for d in glob.glob(path.join(FOLDER_SIMPLE6R, "**", "[0-9]*-NMR-*[0-9]"))
}

xform = (
    lambda s: s.crop(0, 12, inplace=True)
    .erase(1.8, 4, inplace=True)
    .normalize(inplace=True)
)

photo_reaction_dataset = NMRDataset(
    [paths_photo[n] for n in df_photo.name], transform=xform
)
simple2r_reaction_dataset = NMRDataset(
    [paths_simple2r[n] for n in df_simple2r.name],
    target_length=photo_reaction_dataset.min_length,
    transform=xform,
)
simple6r_reaction_dataset = NMRDataset(
    [paths_simple6r[n] for n in df_simple6r.name],
    target_length=photo_reaction_dataset.min_length,
    transform=xform,
)

assert (
    photo_reaction_dataset.min_length
    == simple2r_reaction_dataset.min_length
    == simple6r_reaction_dataset.min_length
)
min_length = photo_reaction_dataset.min_length


@lru_cache(maxsize=None)
def get_reagents(folder, file):
    reactor = file.split("-")[2]
    counter = file.split("-")[0]
    csv_file = path.join(folder, counter, reactor + ".csv")
    if not path.exists(csv_file):
        return ({}, None)
    space = pd.read_csv(csv_file)

    if "photo" in folder:
        volume_first = 2
        volume_second = 2
        volume_third = 2
        volume_base = 2
    elif "Simple" in folder:
        volume_first = space["first_vol"][0]
        volume_second = space["second_vol"][0]
        volume_third = space["third_vol"][0]
        volume_base = 1.5

    volume_acid = 0.5
    volume_cat = 0.5
    volume_tot = volume_first + volume_second + volume_third
    reagents = {}

    reagents[space["first"][0]] = volume_first
    reagents[space["second"][0]] = volume_second

    if "post" in file:
        reagents[space["third"][0]] = volume_third

    if space["base"][0] == True:
        if "Simple" in folder:
            reagents["dbn"] = volume_base
        else:
            reagents["base"] = volume_base
        volume_tot += volume_base
    try:
        if space["acid"][0] == True:
            reagents["acid"] = volume_acid
            volume_tot += volume_acid

        if space["cat"][0] != "none":
            cat = space["cat"][0]
            reagents[cat] = volume_cat
            volume_tot += volume_cat
    except KeyError:
        pass

    return reagents, volume_tot


photo_dirs = glob.glob(path.join(FOLDER_PHOTO_REAGENTS, "*-NMR-0"))
photo_dataset = NMRDataset(photo_dirs, target_length=min_length, transform=xform)
photo_reagents = [path.basename(d).split("-")[0] for d in photo_dirs]
photo_numbers = {d: i for i, d in enumerate(photo_reagents)}

simple_dirs = glob.glob(path.join(FOLDER_SIMPLE_REAGENTS, "*-NMR-0"))
simple_dataset = NMRDataset(simple_dirs, target_length=min_length, transform=xform)
simple_reagents = [path.basename(d).split("-")[0] for d in simple_dirs]
simple_numbers = {d: i for i, d in enumerate(simple_reagents)}
