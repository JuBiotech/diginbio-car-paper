import hagelkorn
import logging
import numpy
import pandas
import pathlib
from typing import Dict, Sequence


DP_DATA = pathlib.Path("..") / "data"
_log = logging.getLogger(__file__)


def fetch_file(fp):
    """Autocompletes a file path to the data directory."""
    fpo = pathlib.Path(fp)
    # Search original path first
    if fpo.exists():
        return fpo

    # Autocomplete to the data directory
    fpd = DP_DATA / fp
    if fpd.exists():
        return fpd

    raise FileNotFoundError(f"The file '{fp}' does not exist and was not found under '{DP_DATA}' either.")


def hagelhash(obj, digits:int=5) -> str:
    """Deterministically calculates a random hagelkorn-style hash of an object."""
    rng = numpy.random.RandomState(hash(obj) % (2**32 - 1))
    alphabet = tuple(hagelkorn.core.DEFAULT_ALPHABET)
    return "".join(rng.choice(alphabet, size=digits))


def get_layout(fp: str, design_cols: Sequence[str]) -> pandas.DataFrame:
    """Loads and validates an experiment layout.
    
    Parameters
    ----------
    fp : path-like
        Path to the XLSX file.
        May be relative to the data folder.
    design_cols : array-like
        Names of columns that describe the experimental design.
        For example: ["pH", "glucose", "iptg"].

    Returns
    -------
    df_layout : pandas.DataFrame
        Layout table with unique culture ID on the index.
    """
    df_layout = pandas.read_excel(fetch_file(fp))

    # Check that is has all required columns
    cols = set(df_layout.columns)
    expected = {"run", "reactor", "assay_well", "product", *design_cols}
    if not cols.issuperset(expected):
        raise ValueError(f"Missing columns from the layout table: {expected - cols}")

    # Remove rows where design information is incomplete
    idrop = [
        i
        for i, row in df_layout.iterrows()
        if numpy.isnan(row["product"]) and not all(row[design_cols].notna())
    ]
    if any(idrop):
        _log.warning(
            "%i rows were dropped because a product concentration was unknown AND information in columns %s was incomplete.",
            len(idrop),
            design_cols
        )
        df_layout = df_layout.drop(idrop)

    # Index it by replicate_id column or generate it
    if not "replicate_id" in cols:
        _log.warning("No 'replicate_id' column found. Generating hagelhashes from run+assay_well.")
        df_layout["replicate_id"] = None
    for row in df_layout.itertuples():
        if pandas.isna(row.replicate_id):
            df_layout.loc[row.Index, "replicate_id"] = hagelhash(str(row.run) + str(row.assay_well))
    df_layout.set_index("replicate_id", inplace=True)
    df_layout = df_layout.astype({
        "run": str,
    })

    # Create unique identifiers for the experiment designs
    design_ids = []
    for _, row in df_layout[design_cols].iterrows():
        vals = tuple([row[d] for d in design_cols])
        if all(~numpy.isnan(vals)):
            design_ids.append(hagelhash(vals))
        else:
            design_ids.append(None)
    df_layout["design_id"] = design_ids
    return df_layout


def read_absorbances(fp) -> pandas.DataFrame:
    """Reads absorbance measurements from TUM-style CSV layout."""
    df = pandas.read_csv(
        fetch_file(fp),
        sep=";",
        index_col=0
    )
    df.rename(inplace=True, columns={
        c : f"{c[0]}{int(c[1:]):02d}"
        for c in df.columns
    })
    df["time_hours"] = (df.index - df.index[0]) / 3600
    return df.set_index("time_hours")


def vectorize_observations(
    df_layout: pandas.DataFrame,
    observations: Dict[str, Dict[str, pandas.DataFrame]],
):
    common = dict(
        index=pandas.Index(df_layout.index.to_numpy(), name="replicate_id"),
        columns=pandas.Index(numpy.arange(len(tuple(observations.values())[0][0])), name="cycle"),
    )
    result_time = pandas.DataFrame(**common)
    result_360 = pandas.DataFrame(**common)
    result_600 = pandas.DataFrame(**common)
    for run, (df360, df600) in observations.items():
        time = df360.index.to_numpy()
        if not numpy.array_equal(df600.index.to_numpy(), time):
            raise ValueError(f"Time vectors for 360 and 600 nm absorbances in run {run} don't match.")
        for row in df_layout[df_layout.run == run].itertuples():
            rid = row.Index
            result_time.loc[rid] = time
            result_360.loc[rid] = df360[row.assay_well].to_numpy()
            result_600.loc[rid] = df600[row.assay_well].to_numpy()
    result_time = result_time.astype(float)
    result_360 = result_360.astype(float)
    result_600 = result_600.astype(float)
    return result_time, result_360, result_600
