import arviz
import calibr8
import copy
import fastprogress
import io
import pymc3
import numpy
import os
from PIL import Image
from typing import Any, Callable, Iterable
from matplotlib import cm, pyplot


def plot_gif(
    fn_plot: Callable[[Any], None],
    fp_out: os.PathLike,
    *,
    data: Iterable[Any],
    fps: int = 3,
    delay_frames: int = 3,
    close: bool = True,
) -> os.PathLike:
    """Create an animated GIF from matplotlib figures.

    Contributors
    ------------
    Michael Osthege <m.osthege@fz-juelich.de>

    Parameters
    ----------
    fn_plot : callable
        A function that takes an element from `data` and creates or updates a current figure.
        Its return value does not matter, but if the same figure is updated
        instead of creating new ones, `close=False` should be set.
    fp_out : path-like
        File name or path for the output.
        See `imageio.get_writer` for possible options other than `*.gif`.
    data : iterable
        An interable over elements that can be passed to `fn_plot`.
    fps : int
        Frames per second for the output.
    delay_frames : int
        Number of frames to append at the end of the GIF.
    close : bool
        If `True` (default) the current matplotlib Figure is closed after every frame.

    Returns
    -------
    fp_out : path-like
        The result file path.
        Use `IPython.display.Image(fp_out)` to display GIFs in Jupyter notebooks.
    """
    frames = []
    for dat in fastprogress.progress_bar(data):
        fn_plot(dat)
        fig = pyplot.gcf()
        with io.BytesIO() as buf:
            # Need the frame without transparency, because it destroys the GIF quality !!
            # Because of https://github.com/matplotlib/matplotlib/issues/14339 PNG always becomes
            # 4-channel, even if transparent=False. Therefore we can just as well skip the compression.
            pyplot.savefig(buf, format="raw", facecolor="w")
            buf.seek(0)
            frame = numpy.frombuffer(buf.getvalue(), dtype="uint8").reshape(
                *fig.canvas.get_width_height()[::-1], 4
            )
            # Because `facecolor="w"` was set, we can cut the alpha channel away.
            frames.append(Image.fromarray(frame[..., :3], "RGB").quantize())
            if close:
                pyplot.close()
    # Repeat the last frame for a delay effect
    frames += [frames[-1]] * delay_frames
    frames[0].save(
        fp_out, save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0
    )
    return fp_out


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
    fig, axs = calibr8.plot_model(cm_600, band_xlim=(0.001, None))
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
        xlim=(0.9, 1.4),
    )
    axs[2].set(
        ylabel="residual",
        xlabel="relative biomass   [-]",
    )
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
            df_A360.loc[t, calibration_wells] - df_A360.iloc[0][calibration_wells],
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
        palette=cm.Greens
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
        palette=cm.Greens
    )
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.A360_of_P.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Blues
    )
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=idata.constant_data.time.values,
        samples=posterior.A360.sel(reaction_well=rwell).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Reds
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
        palette=cm.Blues
    )
    ax.set(
        title="production",
        ylabel="reaction product   [mM]",
        xlabel="time   [h]",
        ylim=(0, None),
    )


    ax = axs[1, 1]
    if "k_mM_per_h" in idata.posterior:
        metric = "k_mM_per_h"
        ylabel = "initial reaction rate   [mM/h]"
    elif "vmax" in idata.posterior:
        metric = "vmax_mM_per_h"
        ylabel = "$v_{max}$   [mM/h]"
    else:
        raise NotImplementedError(f"Did not find a known performance metric in the posterior.")

    violins = ax.violinplot(
        dataset=posterior[metric].T,
        showextrema=False,
        positions=numpy.arange(len(posterior.reaction_well)),
    )
    for i, violin in enumerate(violins['bodies']):
        color = "blue" if posterior.reaction_well[i] == rwell else "grey"
        violin.set_facecolor(color)
        violin.set_edgecolor(color)
    ax.set(
        title="performance metric",
        ylabel=ylabel,
        xlabel="reaction well",
        xticks=numpy.arange(len(posterior.reaction_well)),
        xticklabels=posterior.reaction_well.values,
        ylim=(0, None),
    )
    return
