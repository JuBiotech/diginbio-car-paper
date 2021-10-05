import hagelkorn
import numpy
import pandas
import pathlib
from robotools.transform import make_well_array


DP_DATA = pathlib.Path("..") / "data"


def hagelhash(obj, digits:int=5) -> str:
    """Deterministically calculates a random hagelkorn-style hash of an object."""
    rng = numpy.random.RandomState(hash(obj) % (2**32 - 1))
    alphabet = tuple(hagelkorn.core.DEFAULT_ALPHABET)
    return "".join(rng.choice(alphabet, size=digits))


def get_layout():
    df_layout = pandas.read_excel(DP_DATA / "WellDescription.xlsx").rename(columns={
        "Well": "well",
        "Content": "content",
        "Product, mM": "product",
    })
    df_layout["well"] = make_well_array(R=8, C=12).flatten("C")
    df_layout = df_layout.set_index("well")
    return df_layout


def read_absorbances(fp) -> pandas.DataFrame:
    df = pandas.read_csv(fp, sep=";", index_col=0)
    df.rename(inplace=True, columns={
        c : f"{c[0]}{int(c[1:]):02d}"
        for c in df.columns
    })
    df["time_hours"] = (df.index - df.index[0]) / 3600
    return df.set_index("time_hours")
