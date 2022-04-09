"""
Unit operations of the full analysis pipeline.

Every unit operation requires a working directory argument ``wd``.
Some unit operations need additional kwargs.
"""

import logging
import pathlib

import aesara.tensor as at
import arviz
import calibr8
import mpl_toolkits.axes_grid1
import numpy
import pandas
import pymc as pm
import pyrff
from matplotlib import pyplot
import xarray

import dataloading
import models
import plotting


_log = logging.getLogger(__file__)

DESIGN_COLS = ["iptg", "glucose"]


def plot_experiment_design(wd: pathlib.Path):
    import FigureExperimentalDesign
    FigureExperimentalDesign.ExperimentalDesign(wd)
    return


def plot_btm_overview(wd: pathlib.Path):
    import FigureOverviewProcessData
    FigureOverviewProcessData.btm_overview(wd)
    return


def plot_ph(wd: pathlib.Path):
    import FigureOverviewProcessData
    FigureOverviewProcessData.ph_plot(wd)
    return


def plot_o2(wd: pathlib.Path):
    import FigureOverviewProcessData
    FigureOverviewProcessData.o2_plot(wd)
    return


def load_biomass_calibration(wd: pathlib.Path):
    df_cal = dataloading.read_biomass_calibration().set_index("biomass")
    df_cal.to_excel(wd / "data_biomass_calibration.xlsx")
    return


def fit_biomass_calibration(wd: pathlib.Path, wavelength: int):
    df_cal = pandas.read_excel(wd / "data_biomass_calibration.xlsx", index_col="biomass")
    cm = models.LogisticBiomassAbsorbanceModel()
    calibr8.fit_scipy(
        cm,
        independent=df_cal.index.to_numpy(),
        dependent=df_cal[f"A{wavelength}"].to_numpy(),
        theta_guess=[0, 2, 0.2, 1.5, 0] + [0.01, 0],
        theta_bounds=[
            (-numpy.inf, 0.2), # L_L
            (1.5, numpy.inf),  # L_U
            (-3, 2),           # log10(I_x)
            (0.5, 3),          # dy/dlog10(x)
            (-3, 3),           # c
            (0.0001, 0.1),
            (0, 0.1),
        ],
    )
    cm.save(wd / f"cm_biomass_A{wavelength}.json")
    return


def plot_biomass_calibration(wd: pathlib.Path, wavelength: int):
    cm = models.LogisticBiomassAbsorbanceModel.load(wd / f"cm_biomass_A{wavelength}.json")
    fig, axs = calibr8.plot_model(cm)
    xlabel = r"$\mathrm{Biomass\ [g_{CDW}\ L^{-1}]}$"
    axs[0].set(
        ylabel=r"$\mathrm{Absorbance_{%s\ nm}}\ [-]$" % wavelength,
        xlabel=xlabel,
        xlim=(0, None),
        ylim=(0, None),
    )
    axs[1].set(
        xlabel=xlabel,
        xlim=(0, None),
        ylim=(0, None),
    )
    axs[2].set(
        ylabel=r"$\mathrm{Absolute\ residual\ [-]}$",
        xlabel=xlabel,
        xlim=(0, None),
    )
    plotting.simplified_calibration_legend(axs[1])

    if wavelength == 360:
        plotting.mark_relevant(axs[0], 0.1, 1.3, cm)
        plotting.mark_relevant(axs[1], 0.1, 1.3, cm)
    elif wavelength == 600:
        plotting.mark_relevant(axs[0], 0.1, 0.9, cm)
        plotting.mark_relevant(axs[1], 0.1, 0.9, cm)
    else:
        raise ValueError("Unsupported wavelength.")
    plotting.savefig(fig, f"cm_biomass_A{wavelength}", wd=wd)
    return


def fit_product_calibration(wd: pathlib.Path):
    df = dataloading.read_ph_product_calibration("pH_Kalibrierung_mit_ABAO.xlsx")
    cm = models.LinearProductAbsorbanceModel()
    calibr8.fit_scipy(
        cm,
        independent=df["product"].to_numpy(),
        dependent=df["A360"].to_numpy(),
        theta_guess=[0.2, 0.5, 0.1, 0],
        theta_bounds=[
            (0, 0.5), # intercept
            (0.2, 1),    # slope
            (0.001, 1),  # scale intercept
            (0, 1),      # scale slope
        ],
    )
    cm.save(wd / "cm_product_A360.json")
    return


def plot_product_calibration(wd: pathlib.Path):
    cm = models.LinearProductAbsorbanceModel.load(wd / "cm_product_A360.json")
    fig, axs = calibr8.plot_model(cm)
    xlabel = r"$\mathrm{Product\ [mM]}$"
    axs[0].set(
        ylabel=r"$\mathrm{Absorbance_{360\ nm}}\ [-]$",
        xlabel=xlabel,
        xlim=(0, None),
        ylim=(0, None),
    )
    axs[1].set(
        xlabel=xlabel,
        ylim=(0, None),
    )
    axs[2].set(
        ylabel=r"$\mathrm{Absolute\ residual\ [-]}$",
        xlabel=xlabel,
    )
    plotting.simplified_calibration_legend(axs[1])
    plotting.mark_relevant(axs[0], 0.2, 1.3, cm)
    plotting.mark_relevant(axs[1], 0.2, 1.3, cm)
    plotting.savefig(fig, f"cm_product_A360", wd=wd)
    return


def load_layout(wd: pathlib.Path):
    df_layout = dataloading.get_layout("FullWellDescription.xlsx", DESIGN_COLS)
    df_counts = dataloading.count_replicates(df_layout)
    with pandas.ExcelWriter(wd / "layout.xlsx") as writer:
        df_layout.to_excel(writer, sheet_name="layout")
        df_counts.to_excel(writer, sheet_name="counts")
    return


def load_observations(wd: pathlib.Path):
    df_layout = pandas.read_excel(wd / "layout.xlsx", index_col="replicate_id")
    df_time, df_A360, df_A600 = dataloading.vectorize_observations(
        df_layout,
        observations={
            f"Carboxylase_{v}": (
                dataloading.read_absorbances(rf"Carboxylase_{v}_360nm.csv"),
                dataloading.read_absorbances(rf"Carboxylase_{v}_600nm.csv"),
            )
            for v in [19, 20, 21, 23]
        }
    )
    with pandas.ExcelWriter(wd / "observations.xlsx") as writer:
        df_time.to_excel(writer, sheet_name="time")
        df_A360.to_excel(writer, sheet_name="A360")
        df_A600.to_excel(writer, sheet_name="A600")
    return


def plot_A600_kinetics(wd: pathlib.Path):
    df_layout = pandas.read_excel(wd / "layout.xlsx", sheet_name="layout", index_col="replicate_id")
    df_time = pandas.read_excel(wd / "observations.xlsx", sheet_name="time", index_col="replicate_id")
    df_A600 = pandas.read_excel(wd / "observations.xlsx", sheet_name="A600", index_col="replicate_id")

    fig, _ = plotting.plot_calibration_A600(df_layout, df_A600, df_time)

    plotting.savefig(fig, "plot_A600_kinetics", wd=wd)
    return


def _build_model(wd: pathlib.Path):
    _log.info("Loading calibrations")
    cmX360 = models.LogisticBiomassAbsorbanceModel.load(wd / "cm_biomass_A360.json")
    cmX600 = models.LogisticBiomassAbsorbanceModel.load(wd / "cm_biomass_A600.json")
    cmP360 = models.LinearProductAbsorbanceModel.load(wd / "cm_product_A360.json")
    _log.info("Loading experimental data")
    df_layout = pandas.read_excel(wd / "layout.xlsx", sheet_name="layout", index_col="replicate_id")
    df_time = pandas.read_excel(wd / "observations.xlsx", sheet_name="time", index_col="replicate_id")
    df_A360 = pandas.read_excel(wd / "observations.xlsx", sheet_name="A360", index_col="replicate_id")
    df_A600 = pandas.read_excel(wd / "observations.xlsx", sheet_name="A600", index_col="replicate_id")
    _log.info("Building the model")
    with pm.Model() as pmodel:
        pmodel = models.build_model(
            df_layout,
            df_time,
            df_A360,
            df_A600,
            cmX360,
            cmX600,
            cmP360,
            design_cols=DESIGN_COLS,
        )
    return pmodel


def fit_model(wd: pathlib.Path, **sample_kwargs):
    pmodel = _build_model(wd)
    #_log.info("Creating a model graph")
    #modelgraph = pm.model_to_graphviz(pmodel)
    #_log.info("Saving the model graph")
    #modelgraph.render(filename=str(wd / "model.pdf"), format="pdf")

    sample_kwargs.setdefault("discard_tuned_samples", False)
    sample_kwargs.setdefault("tune", 500)
    sample_kwargs.setdefault("draws", 500)
    # Convergence checks are computed separately
    sample_kwargs.setdefault("compute_convergence_checks", False)

    _log.info("Running MCMC")
    with pmodel:
        idata = pm.sample(**sample_kwargs)
    _log.info("Saving the trace")
    idata.to_netcdf(wd / "trace.nc")
    return


def compute_diagnostics(wd):
    idata = arviz.from_netcdf(wd / "trace.nc")
    # Compute diagnostics only for free variables
    pmodel = _build_model(wd)
    df_diagnostics = arviz.summary(idata, var_names=[rv.name for rv in pmodel.free_RVs])
    df_diagnostics.to_excel(wd / "diagnostics.xlsx")
    return


def check_convergence(wd: pathlib.Path, threshold=1.05):
    df_diagnostics = pandas.read_excel(wd / "diagnostics.xlsx", index_col=0)
    critical = df_diagnostics[df_diagnostics.r_hat > threshold]
    critical_rvs = {rvc.split("[")[0] for rvc in critical.index}
    if critical_rvs:
        raise Exception("The following RVs did not converge: %s", critical_rvs)        
    return


def export_summary(wd: pathlib.Path):
    df_layout = pandas.read_excel(wd / "layout.xlsx", sheet_name="layout", index_col="replicate_id")
    idata = arviz.from_netcdf(wd / "trace.nc")
    summary = plotting.summarize(idata, df_layout)
    summary.to_excel(wd / "summary.xlsx")
    return


def plot_trace(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")
    groups = plotting.interesting_groups(idata.posterior)
    for title, vars in groups.items():
        for key, prefix in [
            ("posterior", "plot_trace"),
            ("warmup_posterior", "plot_warmup"),
        ]:
            if not key in idata:
                _log.warning("InferenceData object has no group %s.", key)
                continue
            _log.info("Plotting %s group %s with variables %s", key, title, vars)
            axs = arviz.plot_trace(idata[key], var_names=vars)
            fig = pyplot.gcf()
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(wd / f"{prefix}_{title}.png")
            pyplot.close()
    return


def plot_posterior_tsne(wd: pathlib.Path):
    import sklearn.manifold

    _log.info("Creating the model")
    pmodel = _build_model(wd)
    _log.info("Loading InferenceData")
    idata = arviz.from_netcdf(wd / "trace.nc")

    _log.info("Flattening InferenceData")
    pst = idata.posterior.stack(sample=("chain", "draw"))
    flat_pst, selectors = plotting.flatten_dataset(
        dataset={
            rv.name : pst[rv.name]
            for rv in pmodel.free_RVs
        },
        skipdim="sample"
    )
    df_samples = flat_pst.to_series().unstack()

    _log.info("Running t-SNE on %i samples with %i features.", *df_samples.to_numpy().shape)
    X = sklearn.manifold.TSNE(
        n_components=2,
        init="pca",
    ).fit_transform(df_samples)

    _log.info("Plotting %i points", len(X))
    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], marker=".", edgecolors="none", s=1)
    ax.set(
        ylabel="t-SNE 1",
        xlabel="t-SNE 2",
        yticks=[],
        xticks=[],
    )
    plotting.savefig(fig, "plot_posterior_tsne", wd=wd)
    return


def plot_prior_vs_posterior(wd: pathlib.Path, var_name: str):
    pmodel = _build_model(wd)
    idata = arviz.from_netcdf(wd / "trace.nc")

    with pmodel:
        prior = pm.sample_prior_predictive(
            var_names=[var_name],
            samples=idata.posterior.sizes["draw"],
        ).prior[var_name]
    posterior = idata.posterior[var_name]


    data = {}
    if "design_dim" in pmodel.RV_dims.get(var_name, []):
        data[f"prior_glucose({var_name})"] = prior.sel(design_dim="glucose")
        data[f"posterior_glucose({var_name})"] = posterior.sel(design_dim="glucose")
        data[f"prior_iptg({var_name})"] = prior.sel(design_dim="iptg")
        data[f"posterior_iptg({var_name})"] = posterior.sel(design_dim="iptg")
    else:
        data[f"prior({var_name})"] = prior
        data[f"posterior({var_name})"] = posterior

    axs = arviz.plot_trace(data)
    xlim = numpy.array([ax.get_xlim() for ax in axs[:, 0]])
    xlim = [min(xlim[:, 0]), max(xlim[:, 1])]
    for ax in axs[:, 0]:
        ax.set_xlim(xlim)
    fig = pyplot.gcf()
    fig.tight_layout()
    plotting.savefig(fig, f"prior_vs_posterior_{var_name}", wd=wd)
    return


def plot_3d_by_design(wd: pathlib.Path, var_name: str):
    idata = arviz.from_netcdf(wd / "trace.nc")
    def fn_plot3d(t):
        plotter = getattr(plotting, f"plot_3d_{var_name}")
        plotter(idata, azim=-45+30*numpy.sin(t*2*numpy.pi))

    plotting.plot_gif(
        fn_plot=fn_plot3d,
        fp_out=wd / f"plot_3d_{var_name}.gif",
        data=numpy.arange(0, 1, 1/60),
        fps=15,
        delay_frames=0
    )
    return


def plot_kinetics(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")
    cmX600 = models.LogisticBiomassAbsorbanceModel.load(wd / "cm_biomass_A600.json")
    summary = pandas.read_excel(wd / "summary.xlsx", index_col="replicate_id")

    def fn_plot(rid):
        plotting.plot_reaction(idata, rid, cm_600=cmX600, reaction_order=summary.index)
        return
    plotting.plot_gif(
        fn_plot=fn_plot,
        fp_out=wd / "plot_kinetics.gif",
        data=summary.index,
        fps=3
    )
    return


def plot_gp_X_factor(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")

    _log.info("Creating the model")
    pmodel = _build_model(wd)

    _log.info("Adding high-resolution GP conditional")
    dense = numpy.linspace(0.01, 6, 300)
    with pmodel:
        _log.info("Adding variables for high-quality predictives")

        if "cycle_segment" not in idata.posterior.coords:
            # The plotting code below only works for models >= 2f12066bcea31f91c26cfe9aac6ec16aeaf58679.
            raise NotImplementedError("This is an outdated InferenceData file!")
        log_X_factor = pmodel.gp_log_X_factor.conditional(
            "dense_log_X_factor",
            Xnew=numpy.log10(dense[:, None]),
            dims="dense_glucose",
        )
        X_factor = pm.Deterministic(
            "dense_X_factor",
            at.exp(log_X_factor),
            dims="dense_glucose",
        )
        X0_design = pm.Deterministic(
            "dense_Xend_2mag",
            pmodel["Xend_batch"] * X_factor,
            dims="dense_glucose",
        )

        _log.info("Sampling posterior predictive")
        _log.info("Sampling prior predictive")
        samples = 1500
        pprior = pm.sample_prior_predictive(
            samples=samples,
            var_names=["Xend_batch", "dense_log_X_factor", "dense_X_factor", "dense_Xend_2mag"],
            return_inferencedata=False,
        )
        _log.info("Sampling posterior predictive")
        thin = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"] / samples)
        pposterior = pm.sample_posterior_predictive(
            idata.posterior.sel(draw=slice(None, None, thin)),
            var_names=["dense_log_X_factor", "dense_X_factor", "dense_Xend_2mag"],
            return_inferencedata=False,
        )
        _log.info("Converting to InferenceData")
        pp = pm.to_inference_data(prior=pprior, posterior_predictive=pposterior)
        del pprior, pposterior

    _log.info("Plotting")
    fig, axs = pyplot.subplots(dpi=200, ncols=2, figsize=(12, 6), sharey=True)

    for ax, ds, letter in zip(axs, [pp.prior, pp.posterior_predictive], "AB"):
        stackdims = ("chain", "draw") if "chain" in ds.dims else ("draw",)
        pm.gp.util.plot_gp_dist(
            ax=ax,
            x=dense,
            samples=ds["dense_Xend_2mag"].stack(sample=stackdims).values.T,
            plot_samples=True,
            palette=pyplot.cm.Greens,
        )
        ax.set(
            xlabel="$\mathrm{Glucose\ feed\ rate}\ \ \ [g_\mathrm{glucose}\ L_\mathrm{reactor}^{-1}\ h^{-1}]}$",
            xlim=(0, max(dense)),
        )
        ax.text(0.02, 0.92, letter, size=24, weight="bold", transform=ax.transAxes)
    axs[0].set(
        ylabel="$X_{end,2mag}\ \ \ [g_\mathrm{CDW}\ L^{-1}]$",
        ylim=(0, 1.5),
    )
    axs[1].set(
        ylim=(0, 1),
    )
    plotting.savefig(fig, "plot_gp_X_factor", wd=wd)
    pyplot.close()
    return


def sample_posterior_predictive_at_design(
    idata,
    pmodel,
    *,
    designs_long: xarray.DataArray,
    dname: str="dense",
    thin: int=1,
) -> arviz.InferenceData:
    dname_id = f"{dname}_id"
    dname_id_glucose = f"{dname}_id_glucose"
    assert tuple(designs_long.dims) == (dname_id, "design_dim")

    _log.info("Adding GP conditional for %i designs", len(designs_long))

    _log.info("Registering new dense coordinates")
    pmodel.add_coord(dname_id, designs_long[dname_id].values)
    pmodel.add_coord(dname_id_glucose, numpy.unique(designs_long.sel(design_dim="glucose").values))

    _log.info("Adding variables for predictives")

    # Predict specific activity at the dense designs
    log_s_design = pmodel.gp_log_s_design.conditional(
        f"{dname}_log_s_design",
        Xnew=designs_long.values,
        dims=dname_id,
        jitter=pm.gp.util.JITTER_DEFAULT
    )
    s_design = pm.Deterministic(f"{dname}_s_design", at.exp(log_s_design), dims=dname_id)

    # Predict fedbatch factors for each glucose design
    long_glucose = designs_long.sel(design_dim="glucose").values
    glucose = pm.ConstantData(f"{dname}_glucose", numpy.unique(long_glucose), dims=dname_id_glucose)
    glucose_log_X_factor = pmodel.gp_log_X_factor.conditional(
        f"{dname}_glucose_log_X_factor",
        Xnew=glucose[:, None],
        dims=dname_id_glucose,
        jitter=pm.gp.util.JITTER_DEFAULT
    )
    glucose_X_factor = pm.Deterministic(
        f"{dname}_glucose_X_factor",
        at.exp(glucose_log_X_factor),
        dims=dname_id_glucose,
    )

    # Subindex into a full-length vector
    idenseglucose_by_design = [
        tuple(glucose.data).index(glc)
        for glc in long_glucose
    ]
    X_factor = pm.Deterministic(
        f"{dname}_X_factor",
        glucose_X_factor[idenseglucose_by_design],
        dims=dname_id
    )

    # Predict rate constants from specific activity and biomass
    k_design = models.predict_k_design(
        X0_fedbatch=pmodel["Xend_batch"],
        fedbatch_factor=X_factor,
        specific_activity=s_design,
        dims=dname_id,
        prefix=f"{dname}_",
    )

    _log.info("Sampling posterior predictive")
    pp = pm.sample_posterior_predictive(
        idata.sel(draw=slice(None, None, thin)),
        var_names=[
            n
            for n, v in pmodel.named_vars.items()
            if n.startswith(dname)
            and v in pmodel.free_RVs + pmodel.deterministics
        ],
        return_inferencedata=False
    )
    _log.info("Saving to InferenceData")
    pposterior = pm.to_inference_data(
        posterior_predictive=pp
    )
    # Include the dense grid in the savefile
    pposterior.posterior_predictive[f"{dname}_long"] = designs_long
    return pposterior


def sample_gp_metric_posterior_predictive(wd: pathlib.Path, n: int=50, thin: int=1):
    idata = arviz.from_netcdf(wd / "trace.nc")

    _log.info("Creating high-resolution designs grid")
    # Create a dense grid
    dense_long = xarray.DataArray(
        models.bounds_to_grid(idata.constant_data.X_design_log10_bounds.values, n),
        dims=("dense_id", "design_dim"),
        coords={
            "dense_id": numpy.arange(n**2),
            "design_dim": idata.posterior.design_dim.values
        }
    )
    dense_grid = models.reshape_dim(
        dense_long,
        from_dim="dense_id",
        to_shape=(n, n),
        to_dims=["dense_design_" + dname for dname in idata.posterior.design_dim.values],
        coords={
            "dense_design_iptg": numpy.unique(dense_long.sel(design_dim="iptg")),
            "dense_design_glucose": numpy.unique(dense_long.sel(design_dim="glucose")),
        }
    )

    _log.info("Creating the model")
    pmodel = _build_model(wd)
    with pmodel:
        pmodel.add_coord("dense_design_iptg", dense_grid.dense_design_iptg.values)
        pmodel.add_coord("dense_design_glucose", dense_grid.dense_design_glucose.values)

        pposterior = sample_posterior_predictive_at_design(
            idata,
            pmodel,
            designs_long=dense_long,
            dname="dense",
            thin=thin,
        )

        pposterior.posterior_predictive["dense_grid"] = dense_grid

        _log.info("Saving to file")
        pposterior.to_netcdf(wd / "predictive_posterior.nc")
    return


def plot_gp_metric_posterior_predictive(
    wd: pathlib.Path,
    var_name="dense_s_design",
):
    label = {
        "dense_s_design": r"$\mathrm{Specific\ activity\ [h^{-1}\ g_{CDW}^{-1}\ L]}$",
        "dense_k_design": r"$\mathrm{Rate\ constant\ [h^{-1}]}$",
    }[var_name]

    idata = arviz.from_netcdf(wd / "trace.nc")
    pposterior = arviz.from_netcdf(wd / "predictive_posterior.nc")

    design_dims = idata.posterior.design_dim.values

    # Extract relevant data arrays
    design_dims = list(idata.constant_data.design_dim.values)
    D = len(design_dims)
    if not D == 2:
        raise NotImplementedError(f"3D visualization for {D}-dimensional designs is not implemented.")
    dense_long = pposterior.posterior_predictive["dense_long"]
    dense_grid = pposterior.posterior_predictive["dense_grid"]
    BOUNDS = numpy.array([
        dense_long.min(dim="dense_id"),
        dense_long.max(dim="dense_id"),
    ]).T

    # Reshape the long-form arrays into the dense 2D grid
    gridshape = tuple(
        dense_grid.sizes["dense_design_" + design_dim]
        for design_dim in design_dims
    )
    Z = models.reshape_dim(
        pposterior.posterior_predictive[var_name],
        from_dim="dense_id",
        to_shape=gridshape,
        to_dims=design_dims,
    )

    # Take the median and HDI of the samples in grid-layout
    median = Z.median(("chain", "draw"))
    hdi = arviz.hdi(Z, hdi_prob=0.9)[var_name]

    assert design_dims[0] == "glucose"
    assert design_dims[1] == "iptg"

    def fn_plot(azim=-65):
        fig = pyplot.figure(dpi=140)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r"$\mathrm{log_{10}(Glucose\ feed\ rate\ [g\ L^{-1}\ h^{-1}])}$")
        ax.set_ylabel(r"$\mathrm{log_{10}(IPTG\ concentration\ [µM])}$")
        ax.set_zlabel(label)

        # Plot surfaces for lower/median/upper
        for q, z in [
            (0.05, hdi.sel(hdi="lower")),
            (0.5, median),
            (0.95, hdi.sel(hdi="higher")),
        ]:
            ax.plot_surface(
                dense_grid.sel(design_dim=design_dims[0]).values,
                dense_grid.sel(design_dim=design_dims[1]).values,
                z.values,
                cmap=pyplot.cm.autumn,
                linewidth=0,
                antialiased=False,
                alpha=1 - abs(q/100 - 0.5) - 0.25
            )
        ax.view_init(elev=25, azim=azim)
        return fig, [[ax]]

    fig, _ = fn_plot(azim=-18)
    plotting.savefig(fig, f"plot_3d_pp_{var_name}", wd=wd)
    pyplot.close()

    def fn_plot3d(t):
        fn_plot(azim=-45+30*numpy.sin(t*2*numpy.pi))
    plotting.plot_gif(
        fn_plot=fn_plot3d,
        fp_out=wd / f"plot_3d_pp_{var_name}.gif",
        data=numpy.arange(0, 1, 1/60),
        fps=15,
        delay_frames=0
    )
    return


def _dense_lookup(pp, feed_rate: float, iptg: float):
    """Look up a dense_id given untransformed process parameters."""

    design_dims = tuple(pp.design_dim.values)
    if not round(numpy.log10(feed_rate), 6) in numpy.round(pp.dense_design_glucose.values, 6):
        raise ValueError(
            f"The selected feed rate of {feed_rate} was not included in the posterior predictive."
            f" Available options are: {10**pp.dense_design_glucose.values}."
        )
    if not round(numpy.log10(iptg), 6) in numpy.round(pp.dense_design_iptg.values, 6):
        raise ValueError(
            f"The selected feed rate of {iptg} was not included in the posterior predictive."
            f" Available options are: {10**pp.dense_design_iptg.values}."
        )

    design_dict = {
        "glucose": numpy.log10(feed_rate),
        "iptg": numpy.log10(iptg),
    }
    design_arr = numpy.array([design_dict[ddim] for ddim in design_dims])
    imatch = numpy.argmin(numpy.abs(numpy.sum(pp.dense_long.values - design_arr, axis=1)))
    dense_id = pp.dense_id.values[imatch]
    numpy.testing.assert_allclose(design_arr, pp.dense_long.sel(dense_id=dense_id))

    return dense_id


def _summarize_median_hdi(var, decimals):
    """Summarize median and 90 % HDI"""
    lower, upper = arviz.hdi(var, hdi_prob=0.9)
    med = numpy.median(var)
    if decimals > 0:
        _round = lambda x: numpy.round(x, decimals)
    else:
        _round = round
    return _round(med), _round(lower), _round(upper)


def predict_units(
    wd: pathlib.Path,
    S0: float=2.5,
    feed_rate: float=4.8,
    iptg: float=32,
    design_log10: dict=None,
):
    """Writes a summary of derived metrics at a particular process design.

    ⚠ The process design must be a subset of the dense posterior predictive prediction.

    Parameters
    ----------
    wd: pathlib.Path
        Current working directory containing the results.
    S0 : float
        Initial 3-hydroxy benzoic acid concentration in the biotransformation.
        Unit: mmol/L
    feed_rate : float
        Glucose feed rate in g/L/h during the expression phase.
    iptg : float
        IPTG concentration in µM.
    design_log10
        Dictionary mapping "iptg" and "glucose" to log10 transformed process design coordinates.
        If provided, this dictionary overrides `feed_rate` and `iptg`.
    """
    if design_log10:
        feed_rate = 10**design_log10["glucose"]
        iptg = 10**design_log10["iptg"]

    postpred = arviz.from_netcdf(wd / "predictive_posterior.nc")
    pp = postpred.posterior_predictive

    # Find the dense_id corresponding to the selected process parameters
    dense_id = _dense_lookup(pp, feed_rate, iptg)

    # Select the rate constant at just that one experimental design
    k_design = pp.dense_k_design.sel(dense_id=dense_id)

    v0, units, volumetric_units = models.to_unit_metrics(S0, k_design)

    summaries = [
        ("Initial reaction rate", *_summarize_median_hdi(v0.values.flatten(), 2), "mmol/h"),
        ("Enzymatic activity", *_summarize_median_hdi(units.values.flatten(), 1), "U"),
        ("Volumetric enzymatic activity", *_summarize_median_hdi(volumetric_units.values.flatten(), 0), "U/mL"),
    ]

    # Write summaries to a text file
    with open(wd / "summary_units.txt", "w", encoding="utf-8") as file:
        line = f"Summarized prediction for\n  {feed_rate} g/L/h glucose feed rate\n  {iptg} µM IPTG\n  {S0} mM 3-hydroxy benzoic acid\n"
        print(line)
        file.write(line)
        for label, med, low, high, unit in summaries:
            line = f"{label} of {med} with 90 % HDI [{low}, {high}] {unit}.\n"
            file.write(line)
            print(line)
    return


def plot_p_best_heatmap(wd: pathlib.Path, ts_seed=None, ts_batch_size=48):
    idata = arviz.from_netcdf(wd / "trace.nc")
    ipp = arviz.from_netcdf(wd / "predictive_posterior.nc")
    pp = ipp.posterior_predictive

    # For each dense design determine the probability that it's the best
    probs = pyrff.sampling_probabilities(pp.dense_k_design.stack(sample=("chain", "draw")), correlated=True)
    probs1d = xarray.DataArray(probs, name="p_best", dims="dense_id")
    best_did = pp.dense_id.values[numpy.argmax(probs)]
    best = pp.dense_long.sel(dense_id=best_did)

    # Reshape into 2D
    probs2d = models.dense_1d_to_2d(probs1d, pp.dense_long)

    # Plot it as a heatmap
    fig, ax = pyplot.subplots(figsize=(5, 5))
    img = plotting.xarrshow(
        ax,
        probs2d.transpose("glucose", "iptg"),
        aspect="auto",
        vmin=0,
    )
    ax.scatter(*idata.constant_data.X_design_log10.values[:,::-1].T, marker="x", color="white")
    ax.scatter(
        [best.sel(design_dim="iptg")],
        [best.sel(design_dim="glucose")],
        marker="o",
        s=100,
        facecolor="none",
        edgecolor="red",
    )
    if ts_seed and ts_batch_size:
        next_dids = pyrff.sample_batch(
            pp.dense_k_design.stack(sample=("chain", "draw")),
            ids=probs1d.dense_id.values,
            correlated=True,
            batch_size=ts_batch_size,
            seed=ts_seed,
        )
        next_x = pp.dense_long.sel(dense_id=list(next_dids))
        ax.scatter(
            next_x.sel(design_dim="iptg"),
            next_x.sel(design_dim="glucose"),
            marker="+",
            color="orange"
        )

    # Draw a colorbar that matches the height of the image
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cbar_kw = dict(
        mappable=img,
        ax=ax,
        cax=divider.append_axes("right", size="5%", pad=0.05),
    )
    cbar = ax.figure.colorbar(**cbar_kw)
    cbar.ax.set_ylabel("$\mathrm{Probability(best)\ [-]}$", rotation=90, va="top")

    ax.set(
        ylabel=r"$\mathrm{log_{10}(Glucose\ feed\ rate\ [g\ L^{-1}\ h^{-1}])}$",
        xlabel=r"$\mathrm{log_{10}(IPTG\ concentration\ [µM])}$",
        title="",
    )
    plotting.savefig(fig, "p_best_k_design", wd=wd)
    return best.to_dataframe().dense_long.to_dict()


def plot_p_best_tested(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")

    pst = idata.posterior.stack(sample=("chain", "draw"))
    probs = pyrff.sampling_probabilities(
        candidate_samples=pst.k_design.values,
        correlated=True,
    )

    x = numpy.arange(len(probs))
    labels = [
        f"{float(glc)} g/L/h\n{float(iptg)} µM"
        for glc, iptg in idata.constant_data.X_design.sel(design_id=pst.k_design.design_id)
    ]

    fig, ax = pyplot.subplots()
    ax.bar(x, probs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set(
        ylabel="p(best tested design)"
    )
    plotting.savefig(fig, "plot_p_best_tested", wd=wd)
    return


def sample_pp_best_tested_vs_predicted(wd: pathlib.Path):
    # Load posterior and posterior predictive
    idata = arviz.from_netcdf(wd / "trace.nc")
    ipp = arviz.from_netcdf(wd / "predictive_posterior.nc")

    # Combine chains
    pst = idata.posterior.stack(sample=("chain", "draw"))
    pp = ipp.posterior_predictive.stack(sample=("chain", "draw"))

    # Select the best tested design and corresponding posterior samples
    pst_probs = pyrff.sampling_probabilities(
        candidate_samples=pst.k_design.values,
        correlated=True,
    )
    i_best_tested = numpy.argmax(pst_probs)
    k_best_tested = pst.k_design.sel(design_id=pst.k_design.design_id[i_best_tested])
    x_best_tested = idata.constant_data.X_design.sel(design_id=k_best_tested.design_id)
    del k_best_tested

    # Select the best predicted design and corresponding posterior samples
    pp_probs = pyrff.sampling_probabilities(
        candidate_samples=pp.dense_k_design.values,
        correlated=True,
    )
    i_best_predicted = numpy.argmax(pp_probs)
    k_best_predicted = pp.dense_k_design.sel(dense_id=pp.dense_k_design.dense_id[i_best_predicted])
    x_best_predicted = (10**pp.dense_long).sel(dense_id=pp.dense_k_design.dense_id[i_best_predicted])
    del k_best_predicted

    # Make a fresh posterior predictive for exactly the two designs.
    # They must be drawn concurrently, because even from the same posterior draws
    # the GP can still cause correllations between the designs.
    pmodel = _build_model(wd)
    best_long = numpy.log10(xarray.concat(
        [x_best_tested, x_best_predicted],
        dim=xarray.IndexVariable("best_id", ["tested", "predicted"])
    ))
    with pmodel:
        pposterior = sample_posterior_predictive_at_design(
            idata,
            pmodel,
            designs_long=best_long,
            dname="best"
        )
        _log.info("Saving to file")
    pposterior.to_netcdf(wd / "predictive_posterior_best.nc")
    return


def report_best_tested_vs_predicted(wd: pathlib.Path):
    pposterior = arviz.from_netcdf(wd / "predictive_posterior_best.nc")

    pp = pposterior.posterior_predictive.best_k_design.stack(sample=("chain", "draw"))
    k_best_tested = pp.sel(best_id="tested")
    k_best_predicted = pp.sel(best_id="predicted")
    _, p_predicted = pyrff.sampling_probabilities(
        candidate_samples=[k_best_tested, k_best_predicted],
        correlated=True
    )

    # Summarize medians and HDIs
    mhdi_best_tested = _summarize_median_hdi(k_best_tested.values, decimals=3)
    mhdi_best_predicted = _summarize_median_hdi(k_best_predicted.values, decimals=3)

    # Write a report
    x_best_tested = 10**pposterior.posterior_predictive.best_long.sel(best_id="tested")
    x_best_predicted = 10**pposterior.posterior_predictive.best_long.sel(best_id="predicted")
    with open(wd / "summary_tested_vs_predicted.txt", "w", encoding="utf-8") as file:
        lines = [
            f"The best tested design was {x_best_tested.to_pandas().to_dict()}.\n",
            f"The best predicted design was {x_best_predicted.to_pandas().to_dict()}.\n",
            f"With a {p_predicted*100:.1f} % probability, the predicted design is better.\n"
            f"\n",
            "Rate constant at the best tested design is {} 1/h with 90 % HDI [{}, {}].\n".format(*mhdi_best_tested),
            "Rate constant at the best predicted design is  {} 1/h with 90 % HDI [{}, {}].\n".format(*mhdi_best_predicted),
        ]
        for line in lines:
            print(line)
            file.write(line)
    
    # Plot the correlation which explains why the probability
    # is so high even though the HDIs overlap.
    ax = arviz.plot_pair(pposterior.posterior_predictive, var_names="best_k_design", figsize=(4,4), kind="hexbin")
    fig = pyplot.gcf()
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    newlims = [
        min(ylims[0], xlims[0]),
        max(ylims[1], xlims[1]),
    ]
    ax.set(
        ylim=newlims,
        xlim=newlims,
    )
    plotting.savefig(fig, "plot_best_k_design_correlation", wd=wd)
    return
