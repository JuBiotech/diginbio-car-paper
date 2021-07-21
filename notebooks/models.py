import logging
import pandas
import calibr8
import aesara.tensor as at
import pymc3
import numpy
import murefi

_log = logging.getLogger(__file__)

class LinearBiomassAbsorbanceModel(calibr8.BasePolynomialModelT):
    def __init__(self, *, independent_key="X", dependent_key="absorbance"):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, mu_degree=1, scale_degree=0, theta_names=["intercept", "slope", "sigma", "df"])


class MichaelisMentenModel(murefi.BaseODEModel):
    def __init__(self):
        self.guesses = dict(S_0=5, P_0=0, v_max=0.1, K_S=1)
        self.bounds = dict(
            S_0=(1, 20),
            P_0=(0, 10),
            v_max=(0.0001, 5),
            K_S=(0.01, 10),
        )
        super().__init__(
            independent_keys=['S', 'P'],
            parameter_names=["S_0", "P_0", "v_max", "K_S"],
        )

    def dydt(self, y, t, theta):
        S, P = y
        v_max, K_S = theta

        dPdt = v_max * S / (K_S + S)
        return [
            -dPdt,
            dPdt,
        ]


def build_model(
    df_layout: pandas.DataFrame,
    df_A360: pandas.DataFrame,
    df_A600: pandas.DataFrame,
    cm_600: calibr8.CalibrationModel,
    kind: str,
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
    cm_600 : calibr8.CalibrationModel
        A calibration model fitted for relative biomass vs. absorbance at 600 nm.
    kind : str
        Which product formation model to apply. One of:
        - "mass action"
        - "michaelis menten"
    """
    reaction_wells = df_layout.loc[numpy.logical_and(
        ~df_layout["content"].isna(),
        df_layout["product"].isna(),
    )].index.to_numpy()
    calibration_wells = df_layout.loc[~df_layout["product"].isna()].index.to_numpy()

    _log.info("Constructing model for %i calibration and %i reaction wells.", len(reaction_wells), len(calibration_wells))

    # Name of dimensions in the model:
    C = "calibration_well"
    R = "reaction_well"
    S = "sampling_cycle"

    with pymc3.Model(coords={
        C: calibration_wells,
        R: reaction_wells,
        S: numpy.arange(len(df_A600)),
    }) as pmodel:
        ################ CALIBRATION MODEL ################
        # This part of the model describes the data-generating process of the absorbances in CALIBRATION wells.
        # We need to include this in the model, because we don't have separate biomass/A360 and biomass/A600 calibrations.

        # The data is ultimately generated from some biomass and product concentrations.
        # We don't know the biomasses in the calibration wells (C) and they change over time (S):
        X_cal = pymc3.Lognormal("X_cal", mu=0, sd=0.1, dims=(S, C))
        # But we have data for the product concentration:
        P_cal = pymc3.Data("P_cal", df_layout.loc[calibration_wells, "product"], dims=C)

        # The absorbance at 360 nm depends on input/response relationships that we don't know.
        # But from an exploratory scatter plot we made guesses 👇 about the slopes.
        A360_per_X = pymc3.Lognormal("A360_per_X", mu=numpy.log(0.6), sd=0.5)
        A360_per_P = pymc3.Lognormal("A360_per_P", mu=numpy.log(1/3), sd=0.5)

        # We don't know how much noise there is in the A360 measurement.
        # We could make this an unknown variable (e.g. σ_A360 ~ HalfNormal(0.05)),
        # but the MCMC algorithms often have a hard time fitting the standard deviation of the likelihood function.
        # So we just assume a conservative 0.05 [a.u.] noise:
        σ_A360 = 0.05

        # The absorbance at 360 nm can be predicted as a function of the concentrations (X_cal, P_cal) and slope parameters.
        A360_of_X_cal = pymc3.Deterministic("A360_of_X_cal", A360_per_X * X_cal, dims=(S, C))
        A360_of_P_cal = pymc3.Deterministic("A360_of_P_cal", A360_per_P * P_cal, dims=C)
        A360_cal = pymc3.Deterministic(
            "A360_cal",
            # To make both compatible, this 👇 adds a broadcastable dimension to product absorbances.
            A360_of_X_cal + A360_of_P_cal[None, :],
            dims=(S, C)
        )
        
        # connect with observations
        obs_cal_A360 = pymc3.Data("obs_cal_A360", df_A360[calibration_wells].to_numpy(), dims=(S, C))
        obs_cal_A600 = pymc3.Data("obs_cal_A600", df_A600[calibration_wells].to_numpy(), dims=(S, C))

        L_cal_A360 = pymc3.Normal("L_of_calibration_A360", mu=A360_cal, sd=σ_A360, observed=obs_cal_A360, dims=(S, C))
        L_cal_A600 = cm_600.loglikelihood(
            x=X_cal,
            y=obs_cal_A600,
            name="L_of_calibration_A600",
            dims=(S, C),
        )
        
        ################ PROCESS MODEL ################
        # This part of the model describes the data-generating process of the absorbances in REACTION wells.
        # Here both X and P are unknown, but the slope parameters from above can be used in the prediction formulas.

        # The product concentration will be a function of the time ⌚.
        time = pymc3.Data("time", df_A600.index.values, dims=S)

        X = pymc3.Lognormal("X", mu=0, sd=0.1, dims=(S, R))
        # The initial substrate concentration is 👇 µM,
        # but we wouldn't be surprised if it was    ~10 % 👇 off.
        S0 = pymc3.Lognormal("S0", mu=numpy.log(2.5), sd=0.1)
        # Instead of modeling an initial product concentration, we can model a time delay
        # since the actual start of the reaction. This way the total amount of substrate/product
        # is preserved and it's a little easier to encode prior knowledge.
        # Here we expect a time delay of about 0.1 hours 👇
        time_delay = pymc3.HalfNormal("time_delay", sd=0.1)
        time_actual = time + time_delay

        if kind == "mass action":
            reaction_half_time = pymc3.HalfNormal("reaction_half_time", 2, dims=R)
            tau = reaction_half_time / numpy.log(2)
            P_yield = pymc3.Deterministic(
                "P_yield",
                (1 - at.exp(-time_actual[:, None] / tau[None, :])),
                dims=(S, R),
            )
            P = pymc3.Deterministic("P", S0 * P_yield, dims=(S, R))
        elif kind == "michaelis menten":
            model = MichaelisMentenModel()
            template = murefi.Replicate()
            n_timesteps = len(pmodel.coords["sampling_cycle"])
            template["P"] = murefi.Timeseries(numpy.arange(n_timesteps), [None]*n_timesteps, independent_key="P", dependent_key="P")
            template["P"].t = time_actual
            P0 = 0
            KS = pymc3.HalfNormal("KS", sd=1)
            vmax = pymc3.Lognormal("vmax", mu=numpy.log(1), sd=1, dims=R)
            P = []
            for r, rwell in enumerate(reaction_wells):
                pred = model.predict_replicate(
                    parameters=[S0, P0, vmax[r], KS],
                    template=template
                )
                P.append(pred["P"].y)
            P = pymc3.Deterministic("P", at.stack(P), dims=(S, R))
        else:
            raise NotImplementedError(f"Invalid model kind '{kind}'.")

        A360_of_X = pymc3.Deterministic("A360_of_X", A360_per_X * X, dims=(S, R))
        A360_of_P = pymc3.Deterministic("A360_of_P", A360_per_P * P, dims=(S, R))
        A360 = pymc3.Deterministic(
            "A360",
            A360_of_X + A360_of_P,
            dims=(S, R)
        )
        
        # connect with observations
        obs_A360 = pymc3.Data("obs_A360", df_A360[reaction_wells].to_numpy(), dims=(S, R))
        obs_A600 = pymc3.Data("obs_A600", df_A600[reaction_wells].to_numpy(), dims=(S, R))

        L_A360 = pymc3.Normal("L_of_reaction_A360", mu=A360, sd=σ_A360, observed=obs_A360, dims=(S, R))
        L_A600 = cm_600.loglikelihood(
            x=X,
            y=obs_A600,
            name="L_of_reaction_A600",
            dims=(S, R),
        )
    return pmodel
