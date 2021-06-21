import arviz
import calibr8
import copy
import pymc3
import numpy
from matplotlib import cm, pyplot
from dibecs_contrib import visualization


def plot_calibration_biomass_observations(df_layout, df_A600):
    fig, ax = pyplot.subplots()

    ax.set_title("product inhibits biomass growth\nbut just a bit")
    groups = list(df_layout.sort_values("product").groupby("product"))
    for g, (concentration, df) in enumerate(groups):
        wells = list(df.index)
        color = cm.autumn(0.1 + g / len(groups))
        label = f"{concentration} mM"
        ax.plot(df_A600.index, df_A600[wells], color=color)
        ax.plot([], [], color=color, label=label)
    ax.legend(frameon=False)
    ax.set(
        xlabel="time   [h]",
        ylabel="absorbance at 600 nm",
        ylim=(0, None),
    )
    pyplot.show()
    return


def plot_cmodel(cm_600):
    cm_600 = copy.deepcopy(cm_600)

    fig, axs = pyplot.subplots(ncols=3, figsize=(12, 4))
    cm_600.cal_independent = numpy.insert(cm_600.cal_independent, 0, 0.001)
    cm_600.cal_dependent = numpy.insert(cm_600.cal_dependent, 0, None)
    calibr8.plot_model(cm_600, fig=fig, axs=axs)
    axs[0].set(
        ylabel="$A_\mathrm{600\ nm}$   [a.u.]",
        xlabel="relative biomass   [-]",
        xlim=(0, None),
        ylim=(0, 0.7),
    )
    axs[1].set(
        ylabel="",
        xlabel="relative biomass   [-]",
        ylim=(0, 0.7),
        xlim=(0.9, 1.5),
    )
    axs[2].set(
        ylabel="residual",
        xlabel="relative biomass   [-]",
        xlim=(0.9, 1.5),
    )
    fig.tight_layout()
    pyplot.show()
    return


def plot_A360_relationships(df_layout, df_A360, df_rel_biomass, calibration_wells):
    fig, axs = pyplot.subplots(ncols=2, figsize=(8, 4))

    ax = axs[0]
    for t in df_A360.index:
        ax.scatter(
            df_layout.loc[calibration_wells, "product"],
            df_A360.loc[t, calibration_wells],
            label=f"t={t:.3f}"
        )
    ax.set(
        ylabel="$A_\mathrm{360\ nm}$   [a.u.]",
        xlabel="product   [mM]",
        title=r"slope $\approx 1/3\ [\frac{a.u.}{mM}]$"
    )
    ax.legend(loc="upper left")

    ax = axs[1]
    for t in df_A360.index:
        ax.scatter(
            df_rel_biomass.loc[t],
            df_A360.loc[t, calibration_wells] - df_A360.loc[0, calibration_wells],
            label=f"t={t:.3f}"
        )
    ax.set(
        ylabel="$\Delta A_\mathrm{360\ nm}$   [a.u.]",
        xlabel="relative biomass  [-]\naccording to A600",
        title=r"slope $\approx 0.6\ [\frac{a.u.}{1}]$"
    )
    ax.legend(loc="upper left")

    fig.tight_layout()
    pyplot.show()


def plot_reaction_well(idata, rwell):
    posterior = idata.posterior.stack(sample=("chain", "draw"))

    fig, axs = pyplot.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle(f"reaction well {rwell}")

    ax = axs[0,0]
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.X.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=visualization.transparentify(cm.Greens)
    )
    ax.set(
        title="biomass change",
        ylabel="posterior(X)   [relative]",
        ylim=(0, None)
    )

    ax = axs[0,1]
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.A360_of_X.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=visualization.transparentify(cm.Greens)
    )
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.A360_of_P.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=visualization.transparentify(cm.Blues)
    )
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.A360.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=visualization.transparentify(cm.Reds)
    )
    ax.scatter(
        idata.constant_data.time.values,
        idata.constant_data.obs_A360.sel(
            reaction_well=list(idata.posterior.reaction_well).index(rwell)
        ).values,
        color="black", marker="x"
    )
    ax.set(
        title="A360 contributions",
        ylabel="$A_{360\ nm}$   [a.u.]",
    )
    ax.legend(handles=[
        ax.scatter([], [], color="black", marker="x", label="observed"),
        ax.fill_between([], [], [], color="red", label="predicted (sum)"),
        ax.fill_between([], [], [], color="green", label="biomass"),
        ax.fill_between([], [], [], color="blue", label="product"),
    ], loc="upper left", frameon=False)

    ax = axs[1, 0]
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.P.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=visualization.transparentify(cm.Blues)
    )
    ax.set(
        title="production",
        ylabel="reaction product   [mM]",
        xlabel="time   [h]",
    )


    ax = axs[1, 1]
    violins = ax.violinplot(
        dataset=posterior.reaction_half_time.T,
        showextrema=False,
        positions=numpy.arange(len(posterior.reaction_well)),
    )
    for i, violin in enumerate(violins['bodies']):
        color = "blue" if posterior.reaction_well[i] == rwell else "grey"
        violin.set_facecolor(color)
        violin.set_edgecolor(color)
    ax.set(
        title="reaction performance",
        ylabel="time to 50 % yield   [h]",
        xlabel="reaction well",
        xticks=numpy.arange(len(posterior.reaction_well)),
        xticklabels=posterior.reaction_well.values,
        ylim=(0, 1.5),
    )
    #pyplot.show()