import aesara
import logging
from typing import Dict, Optional, Sequence, Tuple, Union
import pandas
import calibr8
import aesara.tensor as at
import pymc as pm
import numpy
import xarray


_log = logging.getLogger(__file__)


class LinearBiomassAbsorbanceModel(calibr8.BasePolynomialModelN):
    def __init__(self, *, independent_key="X", dependent_key="absorbance"):
        super().__init__(
            independent_key=independent_key, dependent_key=dependent_key,
            mu_degree=1, sigma_degree=1,
            theta_names=["intercept", "slope", "sigma_i", "sigma_s"]
        )


class LogisticBiomassAbsorbanceModel(calibr8.BaseLogIndependentAsymmetricLogisticN):
    def __init__(self, *, independent_key="biomass", dependent_key="A600"):
        super().__init__(
            independent_key=independent_key, dependent_key=dependent_key,
            sigma_degree=1
        )


class LinearProductAbsorbanceModel(calibr8.BasePolynomialModelN):
    def __init__(self, *, independent_key="P", dependent_key="absorbance"):
        super().__init__(
            independent_key=independent_key, dependent_key=dependent_key,
            mu_degree=1, sigma_degree=1,
            theta_names=["intercept", "slope", "sigma_i", "sigma_s"]
        )


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


def grid_from_coords(
    coords: Dict[str, Sequence[float]],
    prefix: str,
) -> Tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray]:
    """Creates an evenly-spaced grid as the product of coordinate values.

    Parameters
    ----------
    coords : dict
        Coordinate values for each dimension of the tensor.
    prefix : str
        Will be removed from the beginning of dimension names when
        creating "design_dim" coordinate values which refer to the dimensions.

    Returns
    -------
    ids : DataArray
        D-dimensional array of grid point numbering.
        Coords are the ones from the input.
    long : DataArray
        (?, D) shaped array with coordinates of every grid point.
        The length equals the product of dimension lengths.
    grid : DataArray
        (D+1)-dimensional array of grid point coordinate values.
        Coords are the ones from the input plus "design_dim".
    """
    dims = list(coords.keys())
    dims_principled = [dname.replace(prefix, "") for dname in dims]
    lengths = tuple(map(len, coords.values()))
    n = numpy.prod(lengths)
    ids = xarray.DataArray(
        name="dense_ids",
        data=numpy.arange(n).reshape(lengths),
        dims=dims,
        coords=coords,
    )
    stack = ids.stack(dense_id=dims)
    long = xarray.DataArray(
        name="dense_long",
        data=numpy.array([stack[dname] for dname in dims]).T,
        dims=("dense_id", "design_dim"),
        coords={
            "dense_id": stack.values,
            "design_dim": dims_principled,
        }
    )
    grid = reshape_dim(
        long,
        from_dim="dense_id",
        to_shape=lengths,
        to_dims=ids.dims,
        coords=ids.coords
    )
    return ids, long, grid


def reshape_dim(
    var: xarray.DataArray,
    *,
    name: Optional[str]=None,
    from_dim: str,
    to_shape: Sequence[int],
    to_dims: Sequence[str],
    coords: Optional[Dict[str, Sequence]]=None,
) -> xarray.DataArray:
    """Reshapes one of multiple dims of an xarray DataArray.

    Parameters
    ----------
    var : xarray.DataArray
        An xarray array with named dimensions.
    name : str
        A name for the new array. Defaults to ``var.name``.
    from_dim : str
        The name of the dimension that's to be reshaped.
    to_shape : array-like
        Shape into which the ``from_dim`` shall be reshaped.
    to_dims : array-like
        Names of the new dimensions.
    coords : dict, optional
        (New) coordinate values.
        If a dname is not in ``coords``, the ``var.coords``
        is used as a fall-back.  If there's no entry either,
        a ``numpy.arange`` is used as coordinate values.

    Returns
    -------
    arr : xarray.DataArray
        A new data array with the old data in the new shape.
    """
    if not from_dim in var.dims:
        raise Exception(f"Variable has no dimension '{from_dim}'. Dims are: {var.dims}.")
    newshape = []
    newdims = []
    for dim, length in zip(var.dims, var.shape):
        if dim != from_dim:
            newdims.append(dim)
            newshape.append(length)
        else:
            newshape.extend(to_shape)
            newdims.extend(to_dims)
    newcoords = {}
    for dname, dlength in zip(newdims, newshape):
        if dname in (coords or {}):
            newcoords[dname] = coords[dname]
        elif dname in var.coords:
            newcoords[dname] = var.coords[dname]
        else:
            newcoords[dname] = numpy.arange(dlength)
    assert len(newshape) == len(newdims)
    return xarray.DataArray(
        var.values.reshape(*newshape),
        name=name or var.name,
        dims=newdims,
        coords=newcoords
    )


def _add_or_assert_coords(
    coords: Dict[str, Sequence], pmodel: pm.Model
):
    """Ensures that the coords are available in the model."""
    for cname, cvalues in coords.items():
        if cname in pmodel.coords:
            numpy.testing.assert_array_equal(pmodel.coords[cname], cvalues)
        else:
            pmodel.add_coord(name=cname, values=cvalues)


class TidySlices:
    """Maps between dimensions according to the experimental layout."""

    def __init__(self, df_layout: pandas.DataFrame, coords: Dict[str, numpy.ndarray]):
        # by replicate
        self.reactor_by_replicate = [
            coords["reactor_id"].index(df_layout.loc[rid, "reactor_id"])
            for rid in coords["replicate_id"]
        ]

        # by reaction
        self.run_by_reaction = [
            coords["run"].index(df_layout.loc[rid, "run"])
            for rid in coords["reaction"]
        ]
        self.design_by_reaction = [
            coords["design_id"].index(df_layout.loc[rid, "design_id"])
            for rid in coords["reaction"]
        ]
        self.replicate_by_reaction = [
            coords["replicate_id"].index(rid)
            for rid in coords["reaction"]
        ]

        # by reactor_id
        df_reactors = df_layout.drop_duplicates("reactor_id").set_index("reactor_id")
        self.run_by_reactorid = [
            coords["run"].index(df_reactors.loc[rea, "run"])
            for rea in coords["reactor_id"]
        ]
        self.glucose_design_by_reactorid = [
            coords["design_glucose"].index(df_reactors.loc[rea, "glucose"])
            for rea in coords["reactor_id"]
        ]

        # by design_id
        df_designs = df_layout.drop_duplicates("design_id").set_index("design_id")
        self.glucose_by_design = [
            coords["design_glucose"].index(df_designs.loc[did, "glucose"])
            for did in coords["design_id"]
        ]
        super().__init__()


def X_factor_GP(
    ls_mu: float,
    glucose_feed_rates: numpy.ndarray,
    reparameterize: bool=True,
) -> Tuple[at.TensorVariable, pm.gp.Latent]:
    """Creates a 1D Gaussian process modeling a multiplicative biomass factor
    as a function of log10(glucose feed rate).

    Parameters
    ----------
    ls_mu : float
        Approximate lengthscale of fluctuations in the relationship.
        In log10(glucose feed rate [g/L/h]).
    glucose_feed_rates : array-like
        Feed rates in [g/L/h] at which an X_factor should be predicted.

    Returns
    -------
    X_factor : TensorVariable
        Biomass factors predicted from feed rates.
    gp_log_X_factor : pm.gp.Latent
        The underlying Gaussian process describing log(X_factor) as
        a function of log10(glucose feed rate).
    """
    pmodel = pm.modelcontext(None)

    # The factor / glucose relationship hopefully has a sensitivity
    # at around the order of magnitude of our design space.
    ls_X = pm.LogNormal("ls_X", mu=numpy.log(ls_mu), sigma=0.5)

    # Within that design space, the factor possibly varies by ~30 %.
    scaling = pm.LogNormal("scaling_X", mu=numpy.log(0.3), sigma=0.1)

    # Now build the GP for the log-factor:
    mean_func = pm.gp.mean.Zero()
    cov_func = scaling**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls_X)
    gp_log_X_factor = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)

    # Condition the GP on actual glucose feed rates to obtain scaling
    # factors for each unique glucose feed rate.
    # Note that the GP is built on the log10(feed rate) !
    log_X_factor = gp_log_X_factor.prior(
        "log_X_factor",
        at.log10(glucose_feed_rates)[:, None],
        reparameterize=reparameterize,
    )
    X_factor = pm.Deterministic("X_factor", at.exp(log_X_factor), dims="design_glucose")

    # Track dimnames so it shows up in the platemodel
    pmodel.RV_dims["log_X_factor_rotated_"] = ("design_glucose",)
    pmodel.RV_dims["log_X_factor"] = ("design_glucose",)
    pmodel.RV_dims["X_factor"] = ("design_glucose",)
    return X_factor, gp_log_X_factor


def s_design_GP(
    ls_mu: Sequence[float],
    X: numpy.ndarray,
    reparameterize: bool=True,
) -> Tuple[at.TensorVariable, pm.gp.Latent]:
    """Construct a 2-dimensional Gaussian process model to predict specific activity
    from real-valued experimental design.

    Parameters
    ----------
    ls_mu : array-like
        Approximate lengthscales of fluctuations in the relationship.
        Refers to real-valued experimental design.
    X : numpy.ndarray
        Real-valued experimental design for which specific activity should be predicted.

    Returns
    -------
    s_design : TensorVariable
        Predicted specific activity for each experimental design.
    gp_log_s_design_factor : pm.gp.Latent
        The underlying Gaussian process describing log(s_design) as
        a function of real-valued experimental design.
    """
    pmodel = pm.modelcontext(None)

    # Build a GP model of the underlying k, based on glucose and IPTG alone
    ls_s_design = pm.LogNormal('ls_s_design', mu=numpy.log(ls_mu), sigma=0.5, dims="design_dim", initval="moment")

    # The reaction rate k must be strictly positive. So our GP must describe log(k).
    # We expect a k of around log(0.1 mM/h) to log(0.8 mM/h).
    # So the variance of the underlying k(iptg, glucose) function is somewhere around 0.7.
    scaling_s_design = pm.LogNormal('scaling_s_design', mu=numpy.log(0.7), sigma=0.2)

    # Build the 2D GP
    mean_func = pm.gp.mean.Zero()
    cov_func = scaling_s_design**2 * pm.gp.cov.ExpQuad(
        input_dim=2,
        ls=ls_s_design
    )
    gp_log_s_design_factor = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)

    s_design_mean = pm.LogNormal("s_design_mean", mu=numpy.log(0.75), sigma=0.3)
    
    # Now we need to obtain a random variable that describes the k at conditions tested in the dataset.
    log_s_design_factor = gp_log_s_design_factor.prior(
        "log_s_design_factor",
        X=X,
        reparameterize=reparameterize,
    )
    pmodel.RV_dims["log_s_design_factor_rotated_"] = ("design_id",)
    pmodel.RV_dims["log_s_design_factor"] = ("design_id",)
    s_design = pm.Deterministic("s_design", s_design_mean * at.exp(log_s_design_factor), dims="design_id")
    return (s_design, gp_log_s_design_factor)


def build_model(
    df_layout: pandas.DataFrame,
    df_time: pandas.DataFrame,
    df_A360: pandas.DataFrame,
    df_A600: pandas.DataFrame,
    cmX_360: calibr8.CalibrationModel,
    cmX_600: calibr8.CalibrationModel,
    cmP_360: calibr8.CalibrationModel,
    *,
    design_cols: Sequence[str],
    reparameterize:bool = True,
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
        "cycle_segment": numpy.arange(len(df_time.columns) - 1),
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

    i = TidySlices(df_layout, coords)

    # store some of these
    pm.ConstantData("irun_by_reaction", i.run_by_reaction, dims="reaction")
    pm.ConstantData("idesign_by_reaction", i.design_by_reaction, dims="reaction")

    _log.info("Constructing model for %i wells out of which %i are reaction wells.", len(df_layout), len(coords["reaction"]))

    # Track relevant experiment design information and corresponding parameter space metadata as pm.ConstantData containers
    # This information is relevant for GP model components and visualization.
    x_design = df_layout.set_index("design_id")[list(coords["design_dim"])].dropna().drop_duplicates().sort_index().to_numpy()
    X_design = pm.ConstantData("X_design", x_design, dims=("design_id", "design_dim"))
    X_design_log10 = pm.ConstantData("X_design_log10", numpy.log10(x_design), dims=("design_id", "design_dim"))
    del x_design # to keep a single source of truth
    # Bounds of the process parameter space are hard-coded
    # to enable running the model on subsets of all data.
    BOUNDS = numpy.log10([
        [1, 4.8],
        [0.24, 32],
    ])
    SPAN = numpy.ptp(BOUNDS, axis=1)
    pm.ConstantData("X_design_log10_bounds", BOUNDS, dims=("design_dim", "interval"))
    pm.ConstantData("X_design_log10_span", SPAN, dims=("design_dim",))

    # Data containers of unique marginal designs
    X_design_glucose = pm.ConstantData("X_design_glucose", coords["design_glucose"], dims="design_glucose")
    X_design_iptg = pm.ConstantData("X_design_iptg", coords["design_iptg"], dims="design_iptg")

    # The biomass & product concentration will be a function of the time ⌚.
    # Because all kinetics have the same length we can work with a time matrix.
    t = df_time.loc[replicates].to_numpy()
    time = pm.ConstantData("time", t, dims=("replicate_id", "cycle"))
    dt = pm.ConstantData('dt', numpy.diff(t, axis=1), dims=("replicate_id", "cycle_segment"))
    del t

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

    design_idx_glc = pmodel.coords["design_dim"].index("glucose")
    X_factor, pmodel.gp_log_X_factor = X_factor_GP(
        ls_mu=0.2,
        glucose_feed_rates=X_design_glucose,
        reparameterize=reparameterize,
    )

    # Model the biomass story
    # starting from a DASGIP biomass concentration hyperprior
    Xend_batch = pm.LogNormal("Xend_batch", mu=numpy.log(0.5), sigma=0.5)

    # every run may have its own final DASGIP biomass concentration (5 % error)
    Xend_dasgip = pm.LogNormal("Xend_dasgip", mu=at.log(Xend_batch), sigma=0.05, dims="run")
    # final biomasses at the 2mag scale (initial reaction biomasses) follow by multiplication with the feed rate specific factor
    Xend_2mag = pm.Deterministic(
        "Xend_2mag",
        Xend_dasgip[i.run_by_reactorid] * X_factor[i.glucose_design_by_reactorid],
        dims="reactor_id",
    )
    X0_replicate = pm.Deterministic(
        "X0_replicate",
        Xend_2mag[i.reactor_by_replicate],
        dims="replicate_id",
    )

    # Describe biomass growth with a random walk
    # TODO: Double check indexing/slicing to make sure that
    #       1. the coords interpretation matches
    #       2. the GRW doesn't have unidentifiable entries
    #       3. there's no redundant parametrization of the first cycle biomass
    mu_t__diff = pm.Normal(
        'mu_t__diff',
        mu=0, sigma=0.1,
        dims=("replicate_id", "cycle_segment")
    )
    mu_t = pm.Deterministic(
        "mu_t",
        at.cumsum(mu_t__diff, axis=1),
        dims=("replicate_id", "cycle_segment")
    )
    X = pm.Deterministic(
        "X", at.concatenate([
            X0_replicate[:, None],
            X0_replicate[:, None] * at.exp(at.cumsum(mu_t * dt, axis=1)),
        ], axis=1),
        dims=("replicate_id", "cycle"),
    )

    # The initial substrate concentration is 👇 mM,
    # but we wouldn't be surprised if it was    ~10 % 👇 off.
    S0 = pm.LogNormal("S0", mu=numpy.log(2.5), sigma=0.02)

    # But we have data for the product concentration:
    P0 = pm.ConstantData("P0", df_layout.loc[replicates, "product"], dims="replicate_id")

    # Instead of modeling an initial product concentration, we can model a time delay
    # since the actual start of the reaction. This way the total amount of substrate/product
    # is preserved and it's a little easier to encode prior knowledge.
    # Here we expect a time delay of about 0.1 hours 👇
    time_delay = pm.HalfNormal("time_delay", sigma=0.1)
    time_actual = time + time_delay

    s_design, pmodel.gp_log_s_design_factor = s_design_GP(
        ls_mu=0.2,
        X=X_design_log10,
        reparameterize=reparameterize,
    )
    # Unit: [ (1/h) / (g/L) ] 👉 [L/g/h]

    run_effect = pm.LogNormal(
        "run_effect",
        mu=0,
        sigma=0.1,
        dims="run",
        initval=[1] * len(pmodel.coords["run"]),
    )
    k_reaction = pm.LogNormal(
        "k_reaction",
        mu=at.log(
            run_effect[i.run_by_reaction, None] *    # [-]
            s_design[i.design_by_reaction, None] *   # [L/g/h]
            X[i.replicate_by_reaction, :]            # [g/L]
                                                     # 👉 [1/h]
        ),
        sigma=0.05,
        dims=("reaction", "cycle"),
    )

    # Unit of P: [mmol/L]
    P_in_R = pm.Deterministic(
        "P_in_R",
        # mmol/L * (1 - e^(         h              *    1/h    ))
        S0 * (1 - at.exp(-time_actual[mask_RinRID] * k_reaction)),
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
    X_loc, X_scale = cmX_360.predict_dependent(X)
    P_loc, P_scale = cmP_360.predict_dependent(P)
    A360_of_X = pm.Deterministic("A360_of_X", X_loc, dims=("replicate_id", "cycle"))
    A360_of_P = pm.Deterministic("A360_of_P", P_loc, dims=("replicate_id", "cycle"))
    A360 = pm.Deterministic(
        "A360",
        A360_of_X + A360_of_P,
        dims=("replicate_id", "cycle")
    )

    # connect with observations
    pm.ConstantData("obs_A360", obs_A360, dims=("replicate_id", "cycle"))
    obs = pm.ConstantData("obs_A360_notnan", obs_A360[mask_numericA360])
    sigma = pm.Deterministic("sigma", at.sqrt(X_scale**2 + P_scale**2)[mask_numericA360])
    L_A360 = pm.Normal(
        "L_of_A360",
        mu=A360[mask_numericA360],
        sigma=sigma,
        observed=obs
    )

    pm.ConstantData("obs_A600", obs_A600, dims=("replicate_id", "cycle"))
    obs = pm.ConstantData("obs_A600_notnan", obs_A600[mask_numericA600])
    L_cal_A600 = cmX_600.loglikelihood(
        x=X[mask_numericA600],
        y=obs,
        name="L_of_A600",
    )

    # Additionally track an absolute activity metric based on the expected initial biomass concentration (no batch effects)
    predict_k_design(
        X0_fedbatch=Xend_batch,
        fedbatch_factor=X_factor[i.glucose_by_design],
        specific_activity=s_design,
        dims="design_id"
    )
    return pmodel


def predict_k_design(
    *,
    X0_fedbatch: at.TensorVariable,
    fedbatch_factor: at.TensorVariable,
    specific_activity: at.TensorVariable,
    dims: str,
    prefix: str="",
) -> at.TensorVariable:
    """Models a biotransformation rate constant from experiment-independent quantities.

    Parameters
    ----------
    X0_fedbatch
        Initial biomass concentration in the fed-batch phase.
        Unit: [g_CDW / L]
    fedbatch_factor
        Multiplicative factor on the biomass due to the fedbatch strategy.
        Unit: [-]
    specific_activity
        Specific catalytic activity of the whole-cell catalyst at the end of the fed-batch.
        Unit: [(1/h)/(g/L)] (rate constant per g_CDW of catalyst)
    dims
        Dimension names of the symbolic variables.
    prefix : str
        A prefix to prepend to new variable names.

    Returns
    -------
    k_design : at.TensorVariable
        Predicted rate constant.
        Unit: [1/h] (amount of product per biotransformation volume per hour)
    """
    # Design-wise biomass concentrations
    Xend_design = pm.Deterministic(
        prefix + "Xend_design",
        X0_fedbatch * fedbatch_factor,
        dims=dims,
    )

    # Design-wise initial reaction rate
    k_design = pm.Deterministic(
        prefix + "k_design",
        specific_activity * Xend_design,
        dims=dims,
    )
    return k_design


def to_unit_metrics(S0, k, v_catalyst=0.025):
    """Calculates metrix at fixed biotransformation condition.
    
    Parameters
    ----------
    S0
        Initial 3-hydroxy benzoic acid concentration [mmol/L] in the biotransformation.
    k
        Rate constant [1/h] of the whole-cell biocatalyst.
    v_catalyst
        Volume [mL] of the biocatalyst suspension in the biotransformation reaction.

    Returns
    -------
    v0
        Initial reaction rate [mmol/L/h].
    units
        Enzymatic activity [U] = [µmol/L/min].
    volumetric_units
        Enzymatic activity per catalyst volume [U/mL].
    """
    # mM/h = mM * 1/h
    v0 = S0 * k

    #   U = µmol/L/min = mmol/L/h * 1000 mmol/mol / (60 min/h)
    units = v0 * 1000 / 60

    #           U/mL = U / mL
    volumetric_units = units / v_catalyst
    return v0, units, volumetric_units
