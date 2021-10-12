import arviz
import calibr8
import copy
import fastprogress
import io
import mpl_toolkits
import pandas
import pymc as pm
import scipy
import numpy
import os
import xarray
from PIL import Image
from typing import Any, Callable, Dict, Iterable, Optional, Sequence
from matplotlib import cm, pyplot


def plot_absorbance_heatmap(df_layout: pandas.DataFrame, df_360: pandas.DataFrame, df_600: pandas.DataFrame):
    rids = df_layout[df_layout["product"].isna()].index

    fig, axs = pyplot.subplots(nrows=2, dpi=100, figsize=(20, 2), sharex=True)
    ax = axs[0]
    ax.imshow(df_360.loc[rids].to_numpy().T)
    ax.set(
        ylabel="A360\ncycle [-]",
        yticks=[0, 4],
    )

    ax = axs[1]
    ax.imshow(df_600.loc[rids].to_numpy().T)
    ax.set(
        ylabel="A600\ncycle [-]",
        yticks=[0, 4],
        xticks=numpy.arange(len(rids)),
        xticklabels=rids,
        xlabel="replicate ID",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    return fig, axs


def plot_group_kinetics(df_layout, df_time, df_360, df_600, group: str):
    """Makes a dual plot of absorbance time series for one group of replicates.
    Lines are colored by run.
    """
    fig, axs = pyplot.subplots(ncols=2, figsize=(12, 4), dpi=140, sharex=True)

    for ir, run in enumerate(df_layout.run.unique()):
        rids = df_layout[(df_layout["group"] == group) & (df_layout.run == run)].index
        for r, rid in enumerate(rids):
            label = run if r == 0 else None
            axs[0].plot(df_time.loc[rid].T, df_360.loc[rid].T, label=label, color=cm.tab10(ir))
            axs[1].plot(df_time.loc[rid].T, df_600.loc[rid].T, label=label, color=cm.tab10(ir))

    ax = axs[0]
    ax.set(
        ylabel="absorbance at 360 mm",
        xlabel="time   [h]",
        ylim=(0, 2.5),
    )
    ax.legend(loc="upper left", frameon=False)

    ax = axs[1]
    ax.set(
        ylabel="absorbance at 600 mm",
        xlabel="time   [h]",
        ylim=(0, 1),
        xlim=(0, None),
    )
    fig.tight_layout()
    return fig, axs

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


def plot_calibration_A600(df_layout, df_A600, df_time):
    fig, ax = pyplot.subplots(dpi=100)

    groups = list(df_layout.sort_values("product").groupby("product"))
    for g, (concentration, df) in enumerate(groups):
        rids = list(df.index)
        color = cm.autumn(0.1 + g / len(groups))
        label = f"{concentration} mM"
        ax.plot(df_time.loc[rids].T, df_A600.loc[rids].T, color=color)
        ax.plot([], [], color=color, label=label)
    ax.legend(frameon=False)
    ax.set(
        title="A600 kinetics by product concentrations",
        xlabel="time   [h]",
        ylabel="absorbance at 600 nm",
        ylim=(0, None),
    )
    return fig, ax


def plot_calibration_A360(df_layout, df_time, df_A360, df_A600, cm360, cm600, A360_abao=0.212):
    """Makes a scatter plot of 360 nm absorbance of product calibration wells,
    excluding the contributions from biomass and ABAO.
    """
    # For this analysis we only use wells with known product concentration (they got no substrate)
    df = df_layout[~df_layout["product"].isna()]

    X = df_A600.loc[df.index].applymap(cm600.predict_independent)
    A360_biomass = X.applymap(lambda x: cm360.predict_dependent(x)[0])
    
    A360_product = df_A360.loc[df.index] - A360_biomass - A360_abao
    
    fig, axs = pyplot.subplots(dpi=100, figsize=(12, 6), ncols=2, sharey=True)

    ax = axs[0]
    for c, marker in zip(A360_product.columns, "xov1+*D"):
        t = df_time.loc[A360_product.index[0], c]
        ax.scatter(df["product"], A360_product[c], label=f"t={t:.3f}", color=cm.Set1(c), marker=marker)
    ax.set(
        title=r"slope $\approx 0.6\ [\frac{a.u.}{mM}]$",
        ylabel="A360   [a.u.]",
        xlabel="product concentration   [mM]",
    )
    ax.legend(loc="upper left")

    ax = axs[1]
    for p, P in enumerate(df["product"].unique()):
        rids = df[df["product"] == P].index
        x = df_time.loc[rids].T
        y = A360_product.loc[rids].T
        ax.plot(x, y, color=cm.Set1(p))
        ax.plot([], [], label=f"{P:.1f} mM", color=cm.Set1(p))
    ax.set(
        title="A360 increases over time\neven after subtracting biomass & ABAO contribution",
        xlabel="time   [h]",
        ylim=(None, numpy.max(A360_product.values) * 1.2),
    )
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig, ax


def plot_reaction(
    idata,
    rid,
    *,
    ylims=((1.0, 2.5), (2.8, 1.8)),
    cm_600:calibr8.CalibrationModel=None,
    reaction_order: Optional[Sequence[str]]=None,
):
    if reaction_order is None:
        reaction_order = idata.posterior.reaction.values
    reaction_order = list(reaction_order)

    posterior = idata.posterior.stack(sample=("chain", "draw"))
    time = idata.constant_data.time.sel(replicate_id=rid).values

    fig, axs = pyplot.subplots(ncols=2, nrows=2, figsize=(16, 8))
    title = f"reaction {rid}"
    if "X_design" in idata.constant_data:
        reactions = list(posterior.reaction.values)
        idesign = int(idata.constant_data.idesign_by_reaction.values[reactions.index(rid)])
        did = idata.posterior.design_id.values[idesign]
        title += f"\ndesign {did}"
        design_dict = {
            dc : float(idata.constant_data.X_design.sel(
                design_id=did,
                design_dim=dc,
            ))
            for dc in idata.constant_data.design_dim.values
        }
        title += "\n" + ", ".join([f"{k}={v}" for k, v in design_dict.items()])
    fig.suptitle(title)

    ax = axs[0,0]
    if cm_600 is not None:
        loc, scale, df = cm_600.predict_dependent(posterior.X.sel(replicate_id=rid).values.T)
        pm.gp.util.plot_gp_dist(
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
        idata.constant_data.obs_A600.sel(replicate_id=rid),
        marker="x", color="black",
        label="observations"
    )
    ax.set(
        title="A600 contributions",
        ylabel="$A_{600\ nm}$   [a.u.]",
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axs[0,1]
    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.A360_of_X.sel(replicate_id=rid).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Greens
    )
    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.A360_of_P.sel(replicate_id=rid).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Blues
    )
    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=time,
        samples=posterior.A360.sel(replicate_id=rid).values.T,
        plot_samples=False,
        fill_alpha=None,
        palette=cm.Reds
    )
    ax.scatter(
        time,
        idata.constant_data.obs_A360.sel(replicate_id=rid),
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
    pm.gp.util.plot_gp_dist(
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
