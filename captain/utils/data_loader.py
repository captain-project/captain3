import glob
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr

logger = logging.getLogger(__name__)


def load_map(
        filename,
        clip_min: float | None = None,
        clip_max: float | None = None,
        nan_to_num: bool = False,
        select_channel: None | int | range = 0,
):
    extension = os.path.splitext(filename)[1]

    if extension == ".tif":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=rasterio.errors.NotGeoreferencedWarning
            )
            warnings.filterwarnings(
                "ignore",
                message="angle from rectified to skew grid parameter lost in conversion to CF",
            )
            raster = rxr.open_rasterio(filename)
        if isinstance(raster, list):
            raise ValueError(f"{filename} contains several subdatasets; expected one raster")
        data = raster.to_numpy()
        if data.shape[0] > 1 and data.ndim == 3:
            logger.info("%d channels detected in file %s", data.shape[0], filename)
            if select_channel is not None:
                data = data[select_channel]
                logger.info("Selecting channel %s", select_channel)
        name = os.path.basename(filename).split(".tif")[0]
    elif extension == ".npy":
        data = np.load(filename)
        name = os.path.basename(filename).split(".npy")[0]
    else:
        raise ValueError(f"Unrecognized extension: {extension}, {filename}")

    if clip_min is not None or clip_max is not None:
        data = np.clip(data, clip_min, clip_max)
    if nan_to_num:
        np.nan_to_num(data, copy=False)

    return data, name


def load_maps_from_dir(
        dir,
        extension: str = "",
        clip_min: float | None = None,
        clip_max: float | None = None,
):
    filenames = np.sort(glob.glob(os.path.join(dir, "*" + extension)))
    if len(filenames) == 0:
        raise ValueError(f"\nNo files found, check path: \n{dir}\n")
    data = []
    names = []
    for filename in filenames:
        dat, name = load_map(filename)
        data.append(dat)
        names.append(name)

    data = np.squeeze(np.array(data))
    if clip_min is not None or clip_max is not None:
        data = np.clip(data, clip_min, clip_max)

    return data, np.array(names)



def load_trait_table(
        filename,
        species_list: list | np.ndarray,
        ref_column: str | None = None,
        fill_gaps: bool = False,
):
    trait_table = pd.read_csv(filename)
    tbl = reorder_by_species(trait_table, species_list, ref_column)

    if fill_gaps:
        # Calculate means on the subsetted table
        means = tbl.mean(numeric_only=True)
        tbl.fillna(means, inplace=True)

    return tbl


def reorder_by_species(df: pd.DataFrame, species_list, ref_column: str | None = None):
    """
    Reorders and subsets a DataFrame based on a specific list of species names.
    Issues a warning if DataFrame species are excluded, but raises ValueError if
    requested species are missing from the DataFrame.
    """
    if ref_column is None:
        ref_column = df.columns[0]

    # Convert to sets for comparison
    df_species = set(df[ref_column])
    target_species = set(species_list)

    missing_in_df = target_species - df_species
    missing_in_list = df_species - target_species

    # 1. Fatal Error: Requested species don't exist in the DataFrame
    if missing_in_df:
        raise ValueError(
            f"{len(missing_in_df)} species in 'species_list' missing from DataFrame '{ref_column}' column: "
            f"{sorted(missing_in_df, key=str)[:3]} ..."
        )

    # 2. Warning: DataFrame contains extra species (or NaNs) that will be dropped
    if missing_in_list:
        warnings.warn(
            f"{len(missing_in_list)} entries in DataFrame were not found in 'species_list' "
            f"and will be excluded from the subset: {sorted(missing_in_list, key=str)[:3]} ...",
            UserWarning,
            stacklevel=2,
        )

    # 3. Subset and reorder rows to match species_list exactly
    df_reordered = df.set_index(ref_column).reindex(species_list).reset_index()

    return df_reordered


def create_mask_from_map(
        filename: str | Path,
        output_file: str | Path | None = None,
        zero_to_nan: bool = False,
):
    m, _ = load_map(filename)
    if zero_to_nan:
        mask = np.where(np.isfinite(m) & (m != 0), 1.0, np.nan)
    else:
        mask = np.where(np.isfinite(m), 1.0, np.nan)

    if output_file is not None:
        np.save(output_file, mask)
    return mask
