import hagelkorn
import logging
import numpy
import pandas
import pathlib
from robotools.transform import make_well_array


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


def get_layout(fp: str) -> pandas.DataFrame:
    """Loads and validates an experiment layout.
    
    Parameters
    ----------
    fp : path-like
        Path to the XLSX file.
        May be relative to the data folder.

    Returns
    -------
    df_layout : pandas.DataFrame
        Layout table with unique culture ID on the index.
    """
    df_layout = pandas.read_excel(fetch_file(fp))

    # Check that is has all required columns
    cols = set(df_layout.columns)
    expected = {"run", "assay_well", "group", "product"}
    if not cols.issuperset(expected):
        raise ValueError(f"Missing columns from the layout table: {expected - cols}")

    # Index it by replicate_id column or generate it
    if not "replicate_id" in cols:
        _log.warning("No 'replicate_id' column found. Generating hagelhashes from run+assay_well.")
        df_layout["replicate_id"] = [
            hagelhash(str(row.run) + str(row.assay_well))
            for row in df_layout.itertuples()
        ]
    df_layout.set_index("replicate_id", inplace=True)
    df_layout = df_layout.astype({
        "run": str,
    })
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
