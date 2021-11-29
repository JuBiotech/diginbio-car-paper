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


class LinearProductAbsorbanceModel(calibr8.BasePolynomialModelT):
    def __init__(self, *, independent_key="P", dependent_key="absorbance"):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, mu_degree=1, scale_degree=0, theta_names=["intercept", "slope", "sigma", "df"])


class BivariateProductCalibration(calibr8.ContinuousMultivariateModel, calibr8.NormalNoise):
    def __init__(
        self, *,
        independent_key: str="product,pH",
        dependent_key: str="A360",
    ):
        super().__init__(
            independent_key=independent_key,
            dependent_key=dependent_key,
            theta_names="intercept,per_product,I_x,L_max,s,scale".split(","),
            ndim=2,
        )

    @property
    def order(self) -> tuple:
        return tuple(self.independent_key.split(","))

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a student-t-distribution which
        characterizes the dependent variable given values of the independent variable.
        Parameters
        ----------
        x : array-like
            (2,) or (N, 2) values of the independent variables
        theta : optional, array-like
            Parameter vector of the calibration model.

        Returns
        -------
        mu : array-like
            values for the mu parameter of a student-t-distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a student-t-distribution describing the dependent variable
        df : float
            degree of freedom of student-t-distribution
        """
        if theta is None:
            theta = self.theta_fitted
            
        if numpy.ndim(x) == 1:
            x = numpy.atleast_2d(x)
            reduce = True
        else:
            reduce = False
            
        intercept = theta[0]
        per_product = theta[1]
        I_x = theta[2]
        I_y = 1            # to avoid a non-identifiability between I_y and per_product
        Lmax = theta[3]
        s = theta[4]
        scale = theta[5]

        product = x[..., self.order.index("product")]
        pH = x[..., self.order.index("pH")]

        mod = calibr8.core.logistic(pH, [I_x, I_y, Lmax, s])
        
        mu = intercept + product * per_product * mod
        
        if reduce:
            mu = mu[0]
        return mu, scale


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
    cmX_360: calibr8.CalibrationModel,
    cmX_600: calibr8.CalibrationModel,
    cmP_360: calibr8.CalibrationModel,
    *,
    gp_k_design: bool,
    gp_X_factor: bool,
    random_walk_X: bool,
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
    cmX_360 : calibr8.CalibrationModel
        A calibration model fitted for absolute biomass vs. absorbance at 360 nm.
    cmX_600 : calibr8.CalibrationModel
        A calibration model fitted for absolute biomass vs. absorbance at 600 nm.
    cmP_360 : calibr8.CalibrationModel
        A calibration model fitted for absolute ABAO reaction product vs. absorbance at 360 nm.
    gp_k_design : bool
        If `True` a gaussian process model will describe the design-wise specific activity.
        Otherwise design-wise specific activities will be indepdent of each other.
    gp_X_factor : bool
        If `True` the multiplicative effect of glucose feed on initial biomass will be a gaussian process model.
        Otherwise glucose feed specific factors will be indepdent of each other.
    random_walk_X : bool
        If `True` the cycle-wise biomass concentration in each reaction well
        will be described by a Gaussian random walk of cycle-wise growth factor.
        Otherwise the cycle-wise biomasses will be independent of each other.
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
        "reactor_position": df_layout.reactor.astype(str),
        "cycle": df_time.columns.to_numpy(),
        "reactor_id": df_layout["reactor_id"].unique(),
        "reaction": df_layout[df_layout["product"].isna()].index.to_numpy().astype(str),
        "design_id": df_layout[~df_layout["design_id"].isna()].design_id.astype(str),
        "design_dim": design_cols,
        "design_glucose": df_layout["glucose"].dropna().unique(),
        "design_iptg": df_layout["iptg"].dropna().unique(),
        "interval": ("lower", "upper"),
    })
    _add_or_assert_coords(coords, pmodel)
    assert tuple(pmodel.coords["interval"]) == ("lower", "upper")
    coords = pmodel.coords

    # Masking and slicing helper variables
    replicates = list(coords["replicate_id"])
    mask_RinRID = numpy.isin(coords["replicate_id"], coords["reaction"])
    assert len(mask_RinRID) == len(df_layout)
    assert sum(mask_RinRID) == len(coords["reaction"])

    obs_A360 = df_A360.loc[replicates].to_numpy()
    obs_A600 = df_A600.loc[replicates].to_numpy()
    mask_numericA360 = ~numpy.isnan(obs_A360)
    mask_numericA600 = ~numpy.isnan(obs_A360)

    # by replicate
    ireactor_by_replicate = [
        coords["reactor_id"].index(df_layout.loc[rid, "reactor_id"])
        for rid in coords["replicate_id"]
    ]

    # by reaction
    irun_by_reaction = [
        coords["run"].index(df_layout.loc[rid, "run"])
        for rid in coords["reaction"]
    ]
    idesign_by_reaction = [
        coords["design_id"].index(df_layout.loc[rid, "design_id"])
        for rid in coords["reaction"]
    ]
    ireplicate_by_reaction = [
        coords["replicate_id"].index(rid)
        for rid in coords["reaction"]
    ]

    # by reactor_id
    df_reactors = df_layout.drop_duplicates("reactor_id").set_index("reactor_id")
    irun_by_reactorid = [
        coords["run"].index(df_reactors.loc[rea, "run"])
        for rea in coords["reactor_id"]
    ]
    iglucose_design_by_reactorid = [
        coords["design_glucose"].index(df_reactors.loc[rea, "glucose"])
        for rea in coords["reactor_id"]
    ]
    del df_reactors

    # by design_id
    df_designs = df_layout.drop_duplicates("design_id").set_index("design_id")
    iglucose_by_design = [
        coords["design_glucose"].index(df_designs.loc[did, "glucose"])
        for did in coords["design_id"]
    ]
    del df_designs

    # store some of these
    pm.Data("irun_by_reaction", irun_by_reaction, dims="reaction")
    pm.Data("idesign_by_reaction", idesign_by_reaction, dims="reaction")

    _log.info("Constructing model for %i wells out of which %i are reaction wells.", len(df_layout), len(coords["reaction"]))

    # Track relevant experiment design information and corresponding parameter space metadata as pm.Data containers
    # This information is relevant for GP model components and visualization.
    x_design = df_layout.set_index("design_id")[list(coords["design_dim"])].dropna().drop_duplicates().sort_index().to_numpy()
    X_design = pm.Data("X_design", x_design, dims=("design_id", "design_dim"))
    X_design_log10 = pm.Data("X_design_log10", numpy.log10(x_design), dims=("design_id", "design_dim"))
    del x_design # to keep a single source of truth
    BOUNDS = numpy.percentile(X_design_log10.get_value(), [0, 100], axis=0).T
    SPAN = numpy.ptp(BOUNDS, axis=1)
    pm.Data("X_design_log10_bounds", BOUNDS, dims=("design_dim", "interval"))
    pm.Data("X_design_log10_span", SPAN, dims=("design_dim",))

    # Data containers of unique marginal designs
    X_design_glucose = pm.Data("X_design_glucose", coords["design_glucose"], dims="design_glucose")
    X_design_iptg = pm.Data("X_design_iptg", coords["design_iptg"], dims="design_iptg")

    ################ PROCESS MODEL ################
    # The data is ultimately generated from some biomass and product concentrations.
    # We don't know the biomasses in the wells (replicate_id) and they change over time (cycle):
    # TODO: consider biomass prior information from the df_layout

    # We need a biomass model variable for the calculation of a design-wise absolute activity metric.
    # Recap of the biomass story:
    #     process phase:    upstream -------------> expression ---> biotransformation
    #     vessel       :    DASGIP bioreactor ----> 2mag ---------> deep well plate
    #     dimension    :    by run ---------------> by reaction --> by reaction
    # How this can be modeled:
    # + There's a biomass concentration resulting from a glucose feed rate.
    # + That relationship between glucose feed rate and resulting biomass concentration (at the end of the expression phase)
    #   is modeled by a 1-dimensional Gaussian process describing a scaling factor applied to the DASGIP reactor biomass concentration.
    #   The Gaussian process is the log(factor), so centered on 0, making it RFF-approximatable.
    # + Each reaction is a little different, so the glucose-design-wise biomass concentration is used as a hyperprior.

    if gp_X_factor:
        # The factor / glucose relationship hopefully has a sensitivity at around the order of magnitude of our design space.
        ls_X = pm.LogNormal("ls_X", mu=numpy.log(SPAN[coords["design_dim"].index("glucose")]/2), sd=0.1)
        # Within that design space, the factor possibly varies by ~1 order of magnitude.
        scaling = pm.LogNormal('scaling_X', mu=numpy.log(0.3), sd=0.1)
        # Now build the GP for the log-factor:
        mean_func = pm.gp.mean.Zero()
        cov_func = scaling**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls_X)
        pmodel.gp_log_X_factor = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)

        # Condition the GP on actual glucose feed rates to obtain scaling factors for each unique glucose feed rate:
        log_X_factor = pmodel.gp_log_X_factor.prior("log_X_factor", X_design_glucose[:, None], shape=(len(coords["design_glucose"]),))
        X_factor = pm.Deterministic("X_factor", at.exp(log_X_factor), dims="design_glucose")

        # Track dimnames so it shows up in the platemodel
        pmodel.RV_dims["log_X_factor_rotated_"] = ("design_glucose",)
        pmodel.RV_dims["log_X_factor"] = ("design_glucose",)
        pmodel.RV_dims["X_factor"] = ("design_glucose",)
    else:
        X_factor = pm.LogNormal("X_factor", mu=0, sd=0.1, dims="design_glucose")

    # Model the biomass story
    # starting from a DASGIP biomass concentration hyperprior
    X0_base = pm.LogNormal("X0_base", mu=numpy.log(0.5), sd=0.5)

    # For the absolute activity metric we need design-wise biomass concentrations that we can multiply with specific activity
    X0_design = pm.Deterministic(
        "X0_design",
        X0_base * X_factor[iglucose_by_design],
        dims="design_id",
    )

    # every run may have its own final DASGIP biomass concentration (5 % error)
    X0_dasgip = pm.LogNormal("X0", mu=at.log(X0_base), sd=0.05, dims="run")
    # final biomasses at the 2mag scale (initial reaction biomasses) follow by multiplication with the feed rate specific factor
    Xend_2mag = pm.Deterministic(
        "Xend_2mag",
        X0_dasgip[irun_by_reactorid] * X_factor[iglucose_design_by_reactorid],
        dims="reactor_id",
    )
    X0_replicate = pm.Deterministic(
        "X0_replicate",
        Xend_2mag[ireactor_by_replicate],
        dims="replicate_id",
    )[:, None]

    if random_walk_X:
        # Describe biomass growth with a random walk
        # TODO: Double check indexing/slicing to make sure that
        #       1. the coords interpretation matches
        #       2. the GRW doesn't have unidentifiable entries
        #       3. there's no redundant parametrization of the first cycle biomass
        log_dXdc__diff = pm.Normal(
            'log_dXdX__diff_',
            mu=0, sd=0.1,
            dims=("replicate_id", "cycle")
        )
        log_dXdc = pm.Deterministic(
            "log_dXdc",
            at.cumsum(log_dXdc__diff, axis=1),
            dims=("replicate_id", "cycle")
        )
        X = pm.Deterministic(
            "X",
            X0_replicate + X0_replicate * at.exp(log_dXdc),
            dims=("replicate_id", "cycle"),
        )
    else:
        # Cycle-wise biomasses are just centered around the initial biomass hyperprior
        X = pm.LogNormal(
            "X",
            mu=at.repeat(at.log(X0_replicate)[:, None], repeats=len(coords["cycle"]), axis=1),
            sd=0.2,
            dims=("replicate_id", "cycle"),
        )

    # The initial substrate concentration is 👇 mM,
    # but we wouldn't be surprised if it was    ~10 % 👇 off.
    S0 = pm.Lognormal("S0", mu=numpy.log(2.5), sd=0.02)

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

    if not gp_k_design:
        k_design = pm.HalfNormal("k_design", sd=1.5, dims="design_id")
    else:
        # Build a GP model of the underlying k, based on glucose and IPTG alone
        ls = pm.Lognormal('ls', mu=numpy.log(SPAN/2), sd=0.5, dims="design_dim")

        # The reaction rate k must be strictly positive. So our GP must describe log(k).
        # We expect a k of around log(0.1 mM/h) to log(0.8 mM/h).
        # So the variance of the underlying k(iptg, glucose) function is somewhere around 0.7.
        scaling = pm.Lognormal('scaling', mu=numpy.log(0.7), sd=0.2)

        # the literature describes RFFs only for 0 mean !!!
        mean_func = pm.gp.mean.Zero()
        cov_func = scaling**2 * pm.gp.cov.ExpQuad(
            input_dim=len(BOUNDS),
            ls=ls
        )
        pmodel.gp_log_k_design = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        
        # Now we need to obtain a random variable that describes the k at conditions tested in the dataset.
        log_k_design = pmodel.gp_log_k_design.prior(
            "log_k_design",
            X=X_design,
            shape=int(pmodel.dim_lengths["design_id"].eval())
        )
        k_design = pm.Deterministic("k_design", at.exp(log_k_design), dims="design_id")

    run_effect = pm.Lognormal("run_effect", mu=0, sd=0.1, dims="run")
    v_reaction = pm.Lognormal(
        "v_reaction",
        mu=at.log(
            #     [-]          [mM/h/CDW]          [CDW]
            run_effect[irun_by_reaction] * k_design[idesign_by_reaction] * X[ireplicate_by_reaction, 0]
        ),
        sd=0.05,
        dims="reaction"
    )

    P_in_R = pm.Deterministic(
        "P_in_R",
        S0 * (1 - at.exp(-time_actual[mask_RinRID] * v_reaction[:, None])),
        dims=("reaction", "cycle"),
    )

    # Combine fixed & variable P into one tensor
    P = at.empty(
        shape=(pmodel.dim_lengths["replicate_id"], pmodel.dim_lengths["cycle"]),
        dtype=aesara.config.floatX
    )

    P = at.set_subtensor(P[mask_RinRID, :], P_in_R)
    P = at.set_subtensor(P[~mask_RinRID, :], P0[~mask_RinRID, None])
    P = pm.Deterministic("P", P, dims=("replicate_id", "cycle"))

    ################ OBSERVATION MODEL ############
    # The absorbance at 360 nm can be predicted as the sum of ABAO shift, product absorbance and biomass absorbance.

    # Biomass and product contributions
    X_loc, X_scale, X_df = cmX_360.predict_dependent(X)
    P_loc, P_scale, P_df = cmP_360.predict_dependent(P)
    A360_of_X = pm.Deterministic("A360_of_X", X_loc, dims=("replicate_id", "cycle"))
    A360_of_P = pm.Deterministic("A360_of_P", P_loc, dims=("replicate_id", "cycle"))
    A360 = pm.Deterministic(
        "A360",
        A360_of_X + A360_of_P,
        dims=("replicate_id", "cycle")
    )

    # connect with observations
    pm.Data("obs_A360", obs_A360, dims=("replicate_id", "cycle"))
    obs = pm.Data("obs_A360_notnan", obs_A360[mask_numericA360])
    sigma = pm.Deterministic("sigma", at.sqrt(X_scale**2 + P_scale**2)[mask_numericA360])
    L_A360 = pm.StudentT(
        "L_of_A360",
        mu=A360[mask_numericA360],
        # This 👇 calculates the scale as if the distributions were Normal ⚠
        sigma=sigma,
        nu=at.mean([X_df, P_df]),
        observed=obs
    )

    pm.Data("obs_A600", obs_A600, dims=("replicate_id", "cycle"))
    obs = pm.Data("obs_A600_notnan", obs_A600[mask_numericA600])
    L_cal_A600 = cmX_600.loglikelihood(
        x=X[mask_numericA600],
        y=obs,
        name="L_of_A600",
    )

    # Additionally track an absolute activity metric based on the expected initial biomass concentration (no batch effects)
    pm.Deterministic(
        "v_design",
        k_design * X0_design,
        dims="design_id",
    )
    return pmodel
