import arviz
import calibr8
import copy
import fastprogress
import io
import mpl_toolkits
import pandas
import pymc3
import scipy
import numpy
import os
import xarray
from PIL import Image
from typing import Any, Callable, Dict, Iterable, Optional, Sequence
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


def plot_calibration_biomass_observations(df_layout, df_A600, df_time):
    fig, ax = pyplot.subplots()

    ax.set_title("product inhibits biomass growth\nbut just a bit")
    groups = list(df_layout.sort_values("product").groupby("product"))
    for g, (concentration, df) in enumerate(groups):
        rids = list(df.index)
        color = cm.autumn(0.1 + g / len(groups))
        label = f"{concentration} mM"
        ax.plot(df_time.loc[rids].T, df_A600.loc[rids].T, color=color)
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


def plot_A360_relationships(df_layout, df_time, df_A360, df_rel_biomass, calibration_rids):
    fig, axs = pyplot.subplots(ncols=2, figsize=(8, 4))

    ax = axs[0]
    for c in df_A360.columns:
        t = df_time.loc[calibration_rids, c][0]
        ax.scatter(
            df_layout.loc[calibration_rids, "product"],
            df_A360.loc[calibration_rids, c],
            label=f"t={t:.3f}"
        )
    ax.set(
        ylabel="$A_\mathrm{360\ nm}$   [a.u.]",
        xlabel="product   [mM]",
        title=r"slope $\approx 1/3\ [\frac{a.u.}{mM}]$"
    )
    ax.legend(loc="upper left")

    ax = axs[1]
    for c in df_A360.columns:
        t = df_time.loc[calibration_rids, c][0]
        ax.scatter(
            df_rel_biomass.loc[calibration_rids, c],
            df_A360.loc[calibration_rids, c] - df_A360.loc[calibration_rids, 0],
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


def plot_reaction(
    idata,
    rid,
    *,
    ylims=((1.0, 2.5), (2.8, 1.8)),
    cm_600:calibr8.CalibrationModel=None,
    reaction_order: Optional[Sequence[str]]=None,
):
    # 👇 workaround for https://github.com/pymc-devs/pymc/issues/5046
    def get_constant_data(data, dname, cval):
        if not cval in tuple(data[dname].values):
            cval = list(idata.posterior[dname].values).index(cval)
        dat = data.sel({dname: cval}).values
        return dat

    if reaction_order is None:
        reaction_order = idata.posterior.reaction.values
    reaction_order = list(reaction_order)

    posterior = idata.posterior.stack(sample=("chain", "draw"))
    time = get_constant_data(idata.constant_data.time, "replicate_id", rid)

    fig, axs = pyplot.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle(f"reaction {rid}")

    ax = axs[0,0]
    if cm_600 is not None:
        loc, scale, df = cm_600.predict_dependent(posterior.X.sel(replicate_id=rid).values.T)
        pymc3.gp.util.plot_gp_dist(
            ax=ax,
            x=time,
            samples=scipy.stats.t.rvs(loc=loc, scale=scale, df=df),
            plot_samples=False,
            fill_alpha=None,
            palette=cm.Greens,
        )
        ax.fill_between([], [], color="green", label="posterior predictive")

    ax.scatter(
        time,
        get_constant_data(idata.constant_data.obs_A600, "replicate_id", rid),
        marker="x", color="black",
        label="observations"
    )
    ax.set(
        title="A600 contributions",
        ylabel="$A_{600\ nm}$   [a.u.]",
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axs[0,1]
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.A360_of_X.sel(replicate_id=rid).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Greens
    )
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.A360_of_P.sel(replicate_id=rid).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Blues
    )
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.A360.sel(replicate_id=rid).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Reds
    )
    ax.scatter(
        time,
        get_constant_data(idata.constant_data.obs_A360, "replicate_id", rid),
        color="black", marker="x"
    )
    ax.set(
        title="A360 contributions",
        ylabel="$A_{360\ nm}$   [a.u.]",
    )
    ax.legend(handles=[
        ax.scatter([], [], color="black", marker="x", label="observed"),
        ax.fill_between([], [], [], color="red", label="posterior (sum)"),
        ax.fill_between([], [], [], color="green", label="biomass"),
        ax.fill_between([], [], [], color="blue", label="product"),
    ], loc="upper left", frameon=False)

    ax = axs[1, 0]
    pymc3.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.P.sel(replicate_id=rid).values.T,
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
    elif "k_reaction" in idata.posterior:
        metric = "k_reaction"
        ylabel = "initial reaction rate   [mM/h]"
    else:
        raise NotImplementedError(f"Did not find a known performance metric in the posterior.")

    violins = ax.violinplot(
        dataset=posterior[metric].sel(reaction=reaction_order).T,
        showextrema=False,
        positions=numpy.arange(len(reaction_order)),
    )
    for i, violin in enumerate(violins['bodies']):
        color = "blue" if reaction_order[i] == rid else "grey"
        violin.set_facecolor(color)
        violin.set_edgecolor(color)
    ax.set(
        title="performance metric",
        ylabel=ylabel,
        xlabel="replicate id",
        xticks=numpy.arange(len(reaction_order)),
        xticklabels=reaction_order,
        ylim=(0, None),
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    for ax, ylim in zip(axs.flatten(), numpy.array(ylims).flatten()):
        ax.set_ylim(0, ylim)
    return


def plot_reactor_positions(data: Dict[str, xarray.Dataset], df_layout: pandas.DataFrame):
    fig, axs = pyplot.subplots(ncols=len(data), figsize=(8, 6 * len(data)), dpi=200)

    for i, (run, ds) in enumerate(data.items()):
        arr = numpy.zeros((8, 6))
        for rid in ds.reaction.values:
            reactor = df_layout.loc[rid, "reactor"]
            r = "ABCDEFGH".index(reactor[0])
            c = int(reactor[1:]) - 1
            arr[r, c] = numpy.median(ds.sel(reaction=rid))
        
        ax = axs[i]
        im = ax.imshow(arr, vmin=0.7, vmax=1.3)
        # Draw a colorbar that matches the height of the image
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cbar_kw = dict(
            mappable=im,
            ax=ax,
            cax=divider.append_axes("right", size="5%", pad=0.05),
        )
        cbar = ax.figure.colorbar(**cbar_kw)
        cbar.ax.set_ylabel("$k_{reactor} / k_{group}$", rotation=-90, va="bottom")
        ax.set_yticks(numpy.arange(8))
        ax.set_yticklabels("ABCDEFGH")
        ax.set_xticks(numpy.arange(6))
        ax.set_xticklabels([1,2,3,4,5,6])

        ax.set_title(run)
    fig.tight_layout()
    pyplot.show()
    return
