import calibr8
import aesara.tensor as at
import pymc3
import numpy
import murefi


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



    with pymc3.Model(coords={
        "calibration_well": calibration_wells.values,
        "reaction_well": reaction_wells.values,
        "sampling_cycle": numpy.arange(len(df_A600)),
    }) as pmodel:
        ################ CALIBRATION MODEL ################
        X_cal = pymc3.Lognormal("X_cal", mu=0, sd=0.1, dims=("sampling_cycle", "calibration_well"))
        P_cal = pymc3.Data("P_cal", df_layout.loc[calibration_wells, "product"], dims="calibration_well")

        # Prior guesses about the input/response relationships at 360 nm
        A360_per_X = pymc3.Lognormal("A360_per_X", mu=numpy.log(0.6), sd=0.5)
        A360_per_P = pymc3.Lognormal("A360_per_P", mu=numpy.log(3), sd=0.5)
        σ_A360 = 0.05
        
        # Predicting the absorbance at 360 nm
        A360_of_X_cal = pymc3.Deterministic("A360_of_X_cal", A360_per_X * X_cal, dims=("sampling_cycle", "calibration_well"))
        A360_of_P_cal = pymc3.Deterministic("A360_of_P_cal", A360_per_P * P_cal, dims="calibration_well")
        A360_cal = pymc3.Deterministic(
            "A360_cal",
            A360_of_X_cal + A360_of_P_cal[None, :],
            dims=("sampling_cycle", "calibration_well")
        )
        
        # connect with observations
        obs_cal_A360 = pymc3.Data("obs_cal_A360", df_A360[calibration_wells].to_numpy(), dims=("sampling_cycle", "calibration_well"))
        obs_cal_A600 = pymc3.Data("obs_cal_A600", df_A600[calibration_wells].to_numpy(), dims=("sampling_cycle", "calibration_well"))

        L_cal_A360 = pymc3.Normal("L_cal_A360", mu=A360_cal, sd=σ_A360, observed=obs_cal_A360)
        L_cal_A600 = cm_600.loglikelihood(
            x=X_cal,
            y=obs_cal_A600,
            replicate_id="calibration_wells",
            dependent_key=cm_600.dependent_key
        )
        
        ################ PROCESS MODEL ################
        time = pymc3.Data("time", df_A600.index.values, dims="sampling_cycle")
        
        X = pymc3.Lognormal("X", mu=0, sd=0.1, dims=("sampling_cycle", "reaction_well"))
        S0 = pymc3.Lognormal("S0", mu=numpy.log(2.5), sd=0.1)
        P0 = pymc3.HalfNormal("P0", sd=0.1)
        
        reaction_half_time = pymc3.HalfNormal("reaction_half_time", 2, dims="reaction_well")
        tau = reaction_half_time / numpy.log(2)
        P_yield = pymc3.Deterministic(
            "P_yield",
            (1 - at.exp(-time[:, None] / tau[None, :])),
            dims=("sampling_cycle", "reaction_well"),
        )
        P = pymc3.Deterministic("P", S0 * P_yield, dims=("sampling_cycle", "reaction_well"))
        
        A360_of_X = pymc3.Deterministic("A360_of_X", A360_per_X * X, dims=("sampling_cycle", "reaction_well"))
        A360_of_P = pymc3.Deterministic("A360_of_P", A360_per_P * P, dims=("sampling_cycle", "reaction_well"))
        A360 = pymc3.Deterministic(
            "A360",
            A360_of_X + A360_of_P,
            dims=("sampling_cycle", "reaction_well")
        )
        
        # connect with observations
        obs_A360 = pymc3.Data("obs_A360", df_A360[reaction_wells].to_numpy(), dims=("sampling_cycle", "reaction_well"))
        obs_A600 = pymc3.Data("obs_A600", df_A600[reaction_wells].to_numpy(), dims=("sampling_cycle", "reaction_well"))

        L_A360 = pymc3.Normal("L_A360", mu=A360, sd=σ_A360, observed=obs_A360)
        L_A600 = cm_600.loglikelihood(
            x=X,
            y=obs_A600,
            replicate_id="reaction_wells",
            dependent_key=cm_600.dependent_key
        )
    return pmodel
