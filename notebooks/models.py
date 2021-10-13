import aesara
import logging
from typing import Dict, Sequence, Union
import pandas
import calibr8
import aesara.tensor as at
import pymc as pm
import numpy


_log = logging.getLogger(__file__)

class LinearBiomassAbsorbanceModel(calibr8.BasePolynomialModelT):
    def __init__(self, *, independent_key="X", dependent_key="absorbance"):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, mu_degree=1, scale_degree=0, theta_names=["intercept", "slope", "sigma", "df"])


class LogisticBiomassAbsorbanceModel(calibr8.BaseLogIndependentAsymmetricLogisticT):
    def __init__(self, *, independent_key="biomass", dependent_key="A600"):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)



def tidy_coords(
    raw_coords: Dict[str, Sequence[Union[str, int]]]
) -> Dict[str, numpy.ndarray]:
    """Creates a coords dictionary with sorted unique coordinate values."""
    coords = {}
    for dname, rawvals in raw_coords.items():
        coords[dname] = numpy.unique(rawvals)
    return coords


def _add_or_assert_coords(
    coords: Dict[str, Sequence], pmodel: pm.Model
):
    """Ensures that the coords are available in the model."""
    for cname, cvalues in coords.items():
        if cname in pmodel.coords:
            numpy.testing.assert_array_equal(pmodel.coords[cname], cvalues)
        else:
            pmodel.add_coord(name=cname, values=cvalues)


def build_model(
    df_layout: pandas.DataFrame,
    df_time: pandas.DataFrame,
    df_A360: pandas.DataFrame,
    df_A600: pandas.DataFrame,
    cm_360: calibr8.CalibrationModel,
    cm_600: calibr8.CalibrationModel,
    *,
    kind: str,
    design_cols: Sequence[str],
):
    """Constructs the full model for the analysis of one biotransformation experiment.

    Parameters
    ----------
    df_layout : pandas.DataFrame
        Layout of the assay wells.
        index:
        - well: The well ID in the measured MTP
        columns:
        - content: Name of the strain, reactor ID, or condition from which samples originated.
        - product: Product concentration (if known).
    df_A360 : pandas.DataFrame
        Table of absorbance measurements at 360 nm.
        index:
        - time_hours: Time since the beginning of the biotransformation.
        columns:
        - A01,A02,...: Well ID in the measurement MTP.
        values: Measured absorbance for that time/well combination.
    df_A600 : pandas.DataFrame
        Same as df_A360, but for 600 nm.
    cm_360 : calibr8.CalibrationModel
        A calibration model fitted for absolute biomass vs. absorbance at 360 nm.
    cm_600 : calibr8.CalibrationModel
        A calibration model fitted for absolute biomass vs. absorbance at 600 nm.
    kind : str
        Which product formation model to apply. One of:
        - "mass action"
        - "michaelis menten"
    design_cols : array-like
        Names of columns that describe the experimental design.
    """
    pmodel = pm.modelcontext(None)

    assert numpy.array_equal(df_time.index, df_layout.index)
    assert numpy.array_equal(df_A360.index, df_layout.index)
    assert numpy.array_equal(df_A600.index, df_layout.index)

    coords = tidy_coords({
        "run": df_layout.run.astype(str),
        "replicate_id": df_layout.index.to_numpy().astype(str),
        "reactor": df_layout.reactor.astype(str),
        "cycle": df_time.columns.to_numpy(),
        "reaction": df_layout[df_layout["product"].isna()].index.to_numpy().astype(str),
        "design_id": df_layout[~df_layout["design_id"].isna()].design_id.astype(str),
        "design_dim": design_cols,
    })
    _add_or_assert_coords(coords, pmodel)

    # Masking and slicing helper variables
    replicates = list(coords["replicate_id"])
    mask_RinRID = numpy.isin(coords["replicate_id"], coords["reaction"])
    assert len(mask_RinRID) == len(df_layout)
    assert sum(mask_RinRID) == len(coords["reaction"])

    obs_A360 = df_A360.loc[replicates].to_numpy()
    obs_A600 = df_A600.loc[replicates].to_numpy()
    mask_numericA360 = ~numpy.isnan(obs_A360)
    mask_numericA600 = ~numpy.isnan(obs_A360)

    irun_by_reaction = [
        list(coords["run"]).index(df_layout.loc[rid, "run"])
        for rid in coords["reaction"]
    ]
    idesign_by_reaction = [
        list(coords["design_id"]).index(df_layout.loc[rid, "design_id"])
        for rid in coords["reaction"]
    ]
    pm.Data("irun_by_reaction", irun_by_reaction, dims="reaction")
    pm.Data("idesign_by_reaction", idesign_by_reaction, dims="reaction")

    _log.info("Constructing model for %i wells out of which %i are reaction wells.", len(df_layout), len(coords["reaction"]))
    X_design = pm.Data(
        "X_design",
        df_layout.set_index("design_id")[coords["design_dim"]].dropna().drop_duplicates().sort_index().to_numpy(),
        dims=("design_id", "design_dim")
    )
    ################ PROCESS MODEL ################
    # The data is ultimately generated from some biomass and product concentrations.
    # We don't know the biomasses in the wells (replicate_id) and they change over time (cycle):
    # TODO: consider biomass prior information from the df_layout
    X = pm.Lognormal("X", mu=0, sd=0.5, dims=("replicate_id", "cycle"))

    # The initial substrate concentration is 👇 µM,
    # but we wouldn't be surprised if it was    ~10 % 👇 off.
    S0 = pm.Lognormal("S0", mu=numpy.log(2.5), sd=0.05)

    # But we have data for the product concentration:
    P0 = pm.Data("P0", df_layout.loc[replicates, "product"], dims="replicate_id")

    # The product concentration will be a function of the time ⌚.
    # Because all kinetics have the same length we can work with a time matrix.
    time = pm.Data("time", df_time.loc[replicates], dims=("replicate_id", "cycle"))

    # Instead of modeling an initial product concentration, we can model a time delay
    # since the actual start of the reaction. This way the total amount of substrate/product
    # is preserved and it's a little easier to encode prior knowledge.
    # Here we expect a time delay of about 0.1 hours 👇
    time_delay = pm.HalfNormal("time_delay", sd=0.1)
    time_actual = time + time_delay

    if kind == "mass action":
        k_design = pymc3.HalfNormal("k_design", sd=1.5, dims="design_id")

        run_effect = pm.Lognormal("run_effect", mu=0, sd=0.1, dims="run")
        k_reaction = pm.Lognormal(
            "k_reaction",
            mu=at.log([
                run_effect[irun] * k_design[idesign]
                for irun, idesign in zip(irun_by_reaction, idesign_by_reaction)
            ]),
            sd=0.1,
            dims="reaction"
        )

        P_in_R = pm.Deterministic(
            "P_in_R",
            S0 * (1 - at.exp(-time_actual[mask_RinRID] * k_reaction[:, None])),
            dims=("reaction", "cycle"),
        )
    else:
        raise NotImplementedError(f"Invalid model kind '{kind}'.")

    # Combine fixed & variable P into one tensor
    P = at.empty(
        shape=(pmodel.dim_lengths["replicate_id"], pmodel.dim_lengths["cycle"]),
        dtype=aesara.config.floatX
    )

    P = at.set_subtensor(P[mask_RinRID, :], P_in_R)
    P = at.set_subtensor(P[~mask_RinRID, :], P0[~mask_RinRID, None])
    P = pm.Deterministic("P", P, dims=("replicate_id", "cycle"))

    ################ OBSERVATION MODEL ############
    # We don't have a separate calibration model for A360 vs. product concentration.
    # But we can assume linearity and make a guess 👇 for the slope.
    A360_per_P = pm.Lognormal("A360_per_P", mu=numpy.log(1/3), sd=0.5)

    # The absorbance at 360 nm can be predicted as the sum of ABAO shift, product absorbance and biomass absorbance.
    A360_per_ABAOmM = pm.Data("A360_per_ABAOmM", 0.212 / 10)
    ABAO_concentration = pm.Data("ABAO_concentration", 10)
    A360_of_ABAO = pm.Deterministic("A360_of_ABAO", A360_per_ABAOmM * (ABAO_concentration - P), dims=("replicate_id", "cycle"))
    X_loc, X_scale, X_df = cm_360.predict_dependent(X)
    A360_of_X = pm.Deterministic("A360_of_X", X_loc, dims=("replicate_id", "cycle"))
    A360_of_P = pm.Deterministic("A360_of_P", P * A360_per_P, dims=("replicate_id", "cycle"))
    A360 = pm.Deterministic(
        "A360",
        A360_of_ABAO + A360_of_X + A360_of_P,
        dims=("replicate_id", "cycle")
    )
    
    # connect with observations
    pm.Data("obs_A360", obs_A360, dims=("replicate_id", "cycle"))
    obs = pm.Data("obs_A360_notnan", obs_A360[mask_numericA360])
    L_A360 = pm.StudentT(
        "L_of_A360",
        mu=A360[mask_numericA360],
        sigma=X_scale[mask_numericA360],
        nu=X_df,
        observed=obs
    )

    pm.Data("obs_A600", obs_A600, dims=("replicate_id", "cycle"))
    obs = pm.Data("obs_A600_notnan", obs_A600[mask_numericA600])
    L_cal_A600 = cm_600.loglikelihood(
        x=X[mask_numericA600],
        y=obs,
        name="L_of_A600",
    )
    return pmodel
