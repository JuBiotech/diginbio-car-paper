import arviz
import itertools
import pathlib
import calibr8
import fastprogress
import io
import logging
from matplotlib import patches
import mpl_toolkits
import pandas
import pathlib
import pyrff
import sklearn.decomposition
import sklearn.preprocessing
import scipy
import numpy
import os
import re
import xarray
from PIL import Image
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from matplotlib import cm, pyplot, colors

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
DP_RESULTS = DP_ROOT / "results"
DP_RESULTS.mkdir(exist_ok=True)

_log = logging.getLogger(__file__)

pyplot.style.use(DP_ROOT / "notebooks" / "DigInBio.mplstyle")


def _apply_to_first_letter(label: str, action) -> str:
    """Applies a callable `action` to the first non-LaTeX command alphabetic character in a `label`."""
    newlabel = ""
    intex = False
    for i,l in enumerate(label):
        if l == "\\":
            intex = True
            newlabel += l
        elif intex and l == "{":
            intex = False
            newlabel += l
        elif not intex and str.isalpha(l):
            return newlabel + action(l) + label[i+1:]
        else:
            newlabel += l
    return newlabel


def _labelformat_fzj(olabel: Optional[str]) -> str:
    """Applies DIN 461 unit formatting and de-capitalizes the first letter."""
    if not olabel:
        return ""
    label = olabel

    # Find & replace
    label = label.replace("CDW\ [g\ L^{-1}]", "Biomass\ [g_{CDW}\ L^{-1}]")

    # De-capitalize first letter
    label = _apply_to_first_letter(label, str.lower)
    # Exceptions:
    label = label.replace("iPTG", "IPTG")
    label = label.replace("dO", "DO")

    # Reformatting of the unit
    label = re.sub(
        pattern=r"\\ \[(.*?)\]",
        repl="\ /\ {\g<1>}",
        string=label,
    )

    _log.info("Labelconversion FZJ\n\t>>>%s<<<\n\t>>>%s<<<", olabel, label)
    return label


def _labelformat_tum(olabel: Optional[str]) -> str:
    """Uses bracketed unit formatting and capitalizes the first letter."""
    if not olabel:
        return ""
    label = olabel

    # Capitalize first letter
    label = _apply_to_first_letter(label, str.upper)
    # Exceptions:
    label = label.replace("PH", "pH")
    label = label.replace("P(", "p(")

    # NoChange: Original unit formatting is with brackes (easier to RegEx match)
    # NoChange: Biomass vs. CDW wording (original in TUM style)

    _log.info("Labelconversion TUM\n\t>>>%s<<<\n\t>>>%s<<<", olabel, label)
    return label


def apply_fzj_style(fig: pyplot.Figure, orig_labels):
    """Applies the following figure style:
    * First letter small
    * Unit with / notation according to DIN 461
    * No coordinate grid
    * Biomass instead of CDW
    """
    for ax in fig.axes:
        ylabel, xlabel, zlabel = orig_labels[ax]
        # Disable the grid (except for 3D plots)
        if not zlabel:
            ax.grid(False)
        ax.set_ylabel(_labelformat_fzj(ylabel))
        ax.set_xlabel(_labelformat_fzj(xlabel))
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel(_labelformat_fzj(zlabel))
    return


def apply_tum_style(fig: pyplot.Figure, orig_labels):
    """Applies the following figure style:
    * First letter capitalized
    * Unit with [] notation
    * Coordinate grid
    * CDW instead of biomass
    """
    for ax in fig.axes:
        ylabel, xlabel, zlabel = orig_labels[ax]
        # Enable the grid
        ax.grid(True)
        ax.set_ylabel(_labelformat_tum(ylabel))
        ax.set_xlabel(_labelformat_tum(xlabel))
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel(_labelformat_tum(zlabel))
    return


def savefig(fig, name: str, *, wd=DP_RESULTS, **kwargs):
    """Saves a bitmapped and vector version of the figure.
    Parameters
    ----------
    fig
        The figure object.
    name : str
        Filename without extension.
    wd : pathlib.Path
        Directory where to save the figure.
    **kwargs
        Additional kwargs for `pyplot.savefig`.
    """
    orig_labels = {}
    for ax in fig.axes:
        ylabel = ax.get_ylabel()
        xlabel = ax.get_xlabel()
        zlabel = getattr(ax, "get_zlabel", lambda: "")()
        orig_labels[ax] = (ylabel, xlabel, zlabel)

    figure_styles = {
        # Subfolder name : function to apply the style
        "figs_fzj": apply_fzj_style,
        "figs_tum": apply_tum_style,
    }

    for subfolder, apply_style in figure_styles.items():
        _log.debug("Applying figure style for subfolder %s", subfolder)
        apply_style(fig, orig_labels)
        _savefig(fig, name, wd=wd / subfolder, **kwargs)
    return


def _savefig(fig, name: str, *, wd, **kwargs):
    """Internal function used to save figures. See `savefig()`."""
    _log.info("Saving figure '%s' to %s", name, wd)
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("bbox_inches", "tight")
    max_pixels = numpy.array([2250, 2625])
    max_dpi = min(max_pixels / fig.get_size_inches())
    if not "dpi" in kwargs:
        kwargs["dpi"] = max_dpi
    wd.mkdir(exist_ok=True)
    fig.savefig(wd / f"{name}.pdf", **kwargs)
    # Save with & without border to measure the "shrink".
    # This is needed to rescale the dpi setting such that we get max pixels also without the border.
    tkwargs = dict(
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight",
        pad_inches=0.01,
    )
    tkwargs.update(kwargs)
    fp = str(wd / f"{name}.tif")
    fig.savefig(fp, **tkwargs)
    # Measure the size
    actual = numpy.array(pyplot.imread(fp).shape[:2][::-1])
    tkwargs["dpi"] = int(tkwargs["dpi"] * min(max_pixels / actual))
    fig.savefig(fp, **tkwargs)

    img = pyplot.imread(str(wd / f"{name}.tif"))
    pyplot.imsave(str(wd / f"{name}.png"), img)
    return


def to_colormap(dark):
    N = 256
    dark = numpy.array((*dark[:3], 1))
    white = numpy.ones(4)
    cvals = numpy.array([
        (1 - n) * white + n * dark
        for n in numpy.linspace(0, 1, N)
    ])
    # add transparency
    cvals[:, 3] = numpy.linspace(0, 1, N)
    return colors.ListedColormap(cvals)


def transparentify(cmap: colors.Colormap) -> colors.ListedColormap:
    """Creates a transparent->color version from a standard colormap.
    
    Stolen from https://stackoverflow.com/a/37334212/4473230
    
    Testing
    -------
    x = numpy.arange(256)
    fig, ax = pyplot.subplots(figsize=(12,1))
    ax.scatter(x, numpy.ones_like(x) - 0.01, s=100, c=[
        cm.Reds(v)
        for v in x
    ])
    ax.scatter(x, numpy.ones_like(x) + 0.01, s=100, c=[
        redsT(v)
        for v in x
    ])
    ax.set_ylim(0.9, 1.1)
    pyplot.show()
    """
    # Get the colormap colors
    #cm_new = numpy.zeros((256, 4))
    #cm_new[:, :3] = numpy.array(cmap(cmap.N))[:3]
    cm_new = numpy.array(cmap(numpy.arange(cmap.N)))
    cm_new[:, 3] = numpy.linspace(0, 1, cmap.N)
    return colors.ListedColormap(cm_new)


redsT = transparentify(cm.Reds)
greensT = transparentify(cm.Greens)
bluesT = transparentify(cm.Blues)
orangesT = transparentify(cm.Oranges)
greysT = transparentify(cm.Greys)


class FZcolors:
    red = numpy.array((191, 21, 33)) / 255
    green = numpy.array((0, 153, 102)) / 255
    blue = numpy.array((2, 61, 107)) / 255
    orange = numpy.array((220, 110, 0)) / 255


class FZcmaps:
    red = to_colormap(FZcolors.red)
    green = to_colormap(FZcolors.green)
    blue = to_colormap(FZcolors.blue)
    orange = to_colormap(FZcolors.orange)
    black = transparentify(cm.Greys)


def interesting_groups(posterior) -> Dict[str, List[str]]:
    """Get groups of interesting free RV names from the posterior."""
    var_groups = {
        "biomass": [
            "X0_batch",
            "Xend_batch",
            "Xend_dasgip",
            "ls_X,scaling_X,X_factor||X_factor",
            "logdXdc|X",
            "mu_t|X",
        ],
        "biotransformation": [
            "S0",
            "time_delay",
            "ls_s_design,scaling_s_design,s_design||s_design",
            "run_effect",
            "k_reaction",
        ],
    }
    available = tuple(posterior.keys())
    result = {}
    for gname, expressions in var_groups.items():
        selected = []
        for expression in expressions:
            for subset in expression.split("||"):
                names = subset.split(",")
                if set(names).issubset(available):
                    selected += names
                    break
        result[gname] = selected
    return result


def mark_relevant(ax, from_y, upto_y, cm: calibr8.CalibrationModel, as_rect=True):
    """Marks a relevant range in a calibration model."""
    from_x = cm.predict_independent(from_y)
    upto_x = cm.predict_independent(upto_y)
    if as_rect:
        ax.add_patch(
            patches.Rectangle(
                xy=(from_x, from_y),
                width=upto_x - from_x,
                height=upto_y - from_y,
                linewidth=1,
                linestyle="--",
                facecolor="black",
                alpha=0.1,
                zorder=-1000
            )
        )
        return
    xmin, ymin = (ax.transScale + ax.transLimits).transform([from_x, from_y])
    xmax, ymax = (ax.transScale + ax.transLimits).transform([upto_x, upto_y])
    line_kwargs = dict(ls="--", color="black")
    ax.axhline(upto_y, xmin=xmin, xmax=xmax, **line_kwargs)
    ax.axvline(upto_x, ymin=ymin, ymax=ymax, **line_kwargs)


def simplified_calibration_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        labels[0].replace(".0", ""),
        labels[1].replace(".0", "").replace(" likelihood band", ""),
        labels[2].replace(".0", "").replace(" likelihood band", ""),
    ]
    ax.legend(handles=handles, labels=labels, frameon=False)
    return


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
    fig, ax = pyplot.subplots()

    groups = list(df_layout.sort_values("product").groupby("product"))
    for g, (concentration, df) in enumerate(groups):
        rids = list(df.index)
        _log.info("Plotting for RIDs %s", rids)
        color = cm.autumn(0.1 + g / len(groups))
        label = f"{concentration} mM"
        t = df_time.loc[rids].T.to_numpy()
        y = df_A600.loc[rids].T.to_numpy()
        ax.plot(t, y, color=color, marker="x")
        if g == 0:
            label += " 3-hydroxy benzaldehyde"
        ax.plot([], [], color=color, label=label)
    ax.legend(frameon=False)
    ax.set(
        xlabel=r"$\mathrm{Time\ [h]}$",
        ylabel=r"$\mathrm{Absorbance\ at\ 600\ nm\ [a.u.]}$",
        ylim=(0, None),
        xlim=(0, None),
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


def plot_bivariate_calibration(
    df: pandas.DataFrame,
    x1: str, x2:str, z:str,
    *,
    flip: str="",
    cm: calibr8.CalibrationModel=None,
    fig=None,
    ax=None,
):
    """Creates a 3D plot of bivariate calibration points.

    Parameters
    ----------
    df : pandas.DataFrame
        A table with columns corresponding to x/y/z values.
    x1 : str
        Name or part of the name of the column containing the first independent variable.
    x2 : str
        Name or part of the name of the column containing the second independent variable.
    z : str
        Name or part of the name of the column containing the dependent variable.
    flip : str, set, tuple or list
        Axes to flip. This variable is accessed via the `in` operator.
        Valid examples include: `"x1"`, `"x1x2"`, `["x2", "z"]`
    cm : calibr8.CalibrationModel
        A bivariate calibration model. Needs to have a `.order` property and must support
        bivariate independent variables.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    if fig is None:
        fig = pyplot.figure(dpi=140)
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    # plot data points
    col_x = [c for c in df.columns if c in x1][0]
    col_y = [c for c in df.columns if c in x2][0]
    col_z = [c for c in df.columns if c in z][0]
    vals = df[col_z]
    ax.scatter(
        df[col_x], df[col_y], vals,
        color=pyplot.cm.autumn(vals / vals.max()),
        marker='x',
        alpha=1
    )

    # plot calibration model
    if cm:
        ix = cm.order.index(col_x)
        iy = cm.order.index(col_y)
        X = numpy.linspace(cm.cal_independent[:, ix].min(), cm.cal_independent[:, ix].max(), 50)
        Y = numpy.linspace(cm.cal_independent[:, iy].min(), cm.cal_independent[:, iy].max(), 50)
        X, Y = numpy.array(numpy.meshgrid(X, Y))
        x1x2 = numpy.array([
            Y.flatten("F"),
            X.flatten("F"),
        ]).T

        mu, scale = cm.predict_dependent(x1x2)

        qs = [95]
        lowers = []
        uppers = []
        for q in qs:
            lowers.append(cm.scipy_dist.ppf(1-q/100, loc=mu, scale=scale))
            uppers.append(cm.scipy_dist.ppf(q/100, loc=mu, scale=scale))
        Zs = lowers + [mu] + uppers[::-1]
        for q, Z in zip(qs + [50] + qs[::-1], Zs):
            ax.plot_surface(
                X, Y, Z.reshape(50, 50, order="F"),
                cmap=pyplot.cm.autumn,
                linewidth=0,
                antialiased=False,
                alpha=1 - abs(q/100 - 0.5) - 0.25
            )

    if "x1" in flip:
        ax.set_xlim(ax.get_xlim()[::-1])
    if "x2" in flip:
        ax.set_ylim(ax.get_ylim()[::-1])
    if "z" in flip:
        ax.set_zlim(ax.get_zlim()[::-1])

    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    ax.set_zlabel(z)
    return fig, ax


def plot_reaction(
    idata,
    rid,
    *,
    ylims=((1.0, 2.5), (2.8, 1.8)),
    cm_600:calibr8.CalibrationModel=None,
    reaction_order: Optional[Sequence[str]]=None,
    stacked: bool=True,
):
    import pymc as pm

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
        loc, scale = cm_600.predict_dependent(posterior.X.sel(replicate_id=rid).values.T)
        pm.gp.util.plot_gp_dist(
            ax=ax,
            x=time,
            samples=scipy.stats.norm.rvs(loc=loc, scale=scale),
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
        ylabel=r"$\mathrm{A_{600\ nm}\ [a.u.]}$",
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axs[0,1]
    handles = []
    if stacked:
        yup = None
        ydown = numpy.zeros_like(time)
        for label, ds, color in [
            ("biomass", posterior.A360_of_X, "green"),
            ("product", posterior.A360_of_P, "blue"),
        ]:
            yup = ydown + numpy.median(ds.sel(replicate_id=rid).values.T, axis=0)
            handles.append(ax.fill_between(time, ydown, yup, color=color, label=label))
            ydown = yup
    else:
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
            samples=posterior.A360_of_ABAO.sel(replicate_id=rid).values.T,
            plot_samples=False,
            fill_alpha=None,
            palette=cm.Oranges
        )
        handles += [
            ax.fill_between([], [], [], color="red", label="posterior (sum)"),
            ax.fill_between([], [], [], color="green", label="biomass"),
            ax.fill_between([], [], [], color="blue", label="product"),
        ]
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
        ylabel=r"$\mathrm{A_{360\ nm}\ [a.u.]}$",
    )
    ax.legend(
        handles=[
            ax.scatter([], [], color="black", marker="x", label="observed"),
            ax.fill_between([], [], [], color="red", label="posterior (sum)"),
        ] + handles[::-1],
        loc="upper left",
        frameon=False
    )

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
        ylabel=r"$\mathrm{reaction\ product\ [mM]}$",
        xlabel=r"$\mathrm{time\ [h]}$",
        ylim=(0, None),
    )


    ax = axs[1, 1]
    metric = "k_reaction"
    ylabel = r"$\mathrm{rate\ constant\ [1/h]}$"

    x = posterior[metric]
    if "cycle" in x.coords:
        # Consider cycle-wise metrics only in cycle 0.
        x = x.sel(cycle=0)
    violins = ax.violinplot(
        dataset=x.sel(reaction=reaction_order).T,
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


def plot_3d_s_design(idata, azim=-65):
    return plot_3d_by_design(idata, "s_design", azim=azim, label="specific activity\n$\mathrm{k_{design}\ [h^{-1} / (g_{CDW}\ h^{-1})]}$")


def plot_3d_k_design(idata, azim=-65):
    return plot_3d_by_design(idata, "k_design", azim=azim, label="rate constant\n$v_{design}\ [h^{-1}]$")


def plot_3d_by_design(idata, var_name: str, *, label: str, azim=-65):
    # Extract relevant data arrays
    design_dims = list(idata.constant_data.design_dim.values)
    X = numpy.log10(idata.constant_data.X_design.sel(design_dim=design_dims))
    Z = idata.posterior[var_name]

    D = len(design_dims)
    BOUNDS = numpy.array([
        X.min(dim="design_id"),
        X.max(dim="design_id"),
    ]).T
    if not D == 2:
        raise NotImplementedError(f"3D visualization for {D}-dimensional designs is not implemented.")

    fig = pyplot.figure(dpi=140)
    ax = fig.add_subplot(111, projection='3d')

    # Plot a basic wireframe.
    ax.set_xlabel(f'log10({design_dims[0]})')
    ax.set_ylabel(f'log10({design_dims[1]})')
    ax.set_zlabel(label)

    # plot observations
    x = X.values
    z = Z.median(dim=("chain", "draw"))
    hdi = arviz.hdi(Z, hdi_prob=0.9)[var_name]
    zerr = numpy.abs(hdi - z)
    ax.errorbar(
        x[:, 0], x[:, 1], z,
        zerr=zerr.T,
        fmt=" ",
        ecolor=cm.autumn(z / z.max())
    )
    ax.scatter(
        x[:,0], x[:,1], z,
        color=cm.autumn(z / z.max()),
        marker='x',
        alpha=1
    )
    ax.view_init(elev=25, azim=azim)
    return fig, ax


def p_best_dataarray(var) -> xarray.DataArray:
    """Applies pyrff.sampling_probabilities on posterior samples, maintaining coords."""
    samples = var.stack(sample=("chain", "draw"))
    dim = samples.dims[0]
    return xarray.DataArray(
        pyrff.sampling_probabilities(samples.values, correlated=True),
        dims=(dim,),
        coords={dim: var[dim]}
    )


def summarize(idata, df_layout) -> pandas.DataFrame:
    def med_hdi(samples, ci_prob=0.9):
        hdi = arviz.hdi(samples, hdi_prob=ci_prob)
        name = tuple(hdi.data_vars)[0]
        lower, upper = hdi[name]
        return float(lower), float(numpy.median(samples)), float(upper)

    df = df_layout.loc[idata.posterior.reaction.values]

    # Add columns with probability of being the best
    df["p_best_design"] = p_best_dataarray(idata.posterior.s_design).sel(design_id=list(df.design_id)).values

    import models
    v0, units, volumetric_units = models.to_unit_metrics(S0=2.5, k=idata.posterior.k_design)

    for name, var, coord in [
        ("s_design_1/h_per_gCDW/L", idata.posterior.s_design, "design_id"),
        ("k_design_1/h", idata.posterior.k_design, "design_id"),
        ("k_reaction_1/h", idata.posterior.k_reaction, "reaction"),
        ("v0_mmol/L/h", v0, "design_id"),
        ("units_U", units, "design_id"),
        ("volumetric_units_U/mL", volumetric_units, "design_id"),
    ]:
        df[name + "_lower"] = None
        df[name] = None
        df[name + "_upper"] = None

        if "cycle" in var.coords:
            # Newer model versions have cycle-wise k_reaction.
            # Here we're just interested in the first cycle.
            var = var.sel(cycle=0)

        for rid in df.index:
            if coord == "reaction":
                cval = rid
            else:
                cval = df.loc[rid, coord]
            df.loc[rid, [name + "_lower", name, name + "_upper"]] = med_hdi(
                var.sel({coord : cval})
            )
            
        assert numpy.all(df[name + "_lower"] < df[name])
        assert numpy.all(df[name + "_upper"] > df[name])
    df = df.sort_values("s_design_1/h_per_gCDW/L")
    return df


def xarrshow(ax, xarr: xarray.DataArray, **kwargs):
    """Plots a DataArray as a heatmap and labels it.
    
    Parameters
    ----------
    ax
        Matplotlib Axes to use.
    xarr
        A 2-dimensional data array.
    **kwargs
        Will be forwarded to `ax.imshow`.
    
    Returns
    -------
    img
        The return value of `ax.imshow`.
    """
    assert xarr.ndim == 2

    dv, dh = xarr.dims
    dx = numpy.diff(xarr[dh].values)[0]
    dy = numpy.diff(xarr[dv].values)[0]

    extent = [
        min(xarr[dh].values) - dx/2,
        max(xarr[dh].values) + dx/2,
        max(xarr[dv].values) + dy/2,
        min(xarr[dv].values) - dy/2,
    ]
    kwargs.setdefault("extent", extent)

    img = ax.imshow(xarr, **kwargs)
    ax.set(
        ylabel=dv,
        xlabel=dh,
        title=xarr.name,
    )
    return img


def flatten_dataset(
    dataset: xarray.Dataset,
    skipdim:str="sample"
) -> Tuple[xarray.DataArray, Dict[str, Tuple[str, Dict[str, Any]]]]:
    """Flatten all dimensions except one that's common across the entire dataset.
    
    Parameters
    ----------
    dataset
        An xarray Dataset where all variables have at least one dimension in common.
        For example: "sample".
    skipdim
        Name of the shared dimension that shall not be flattened away.
        
    Returns
    -------
    flattened
        An xarray DataArray with dims ``(skipdim, "dimension")`` where
        the "dimension" indicates the name of the original variable and the
        coordinate values at which it was selected.
    selectors
        A dictionary of tuples, where keys are the coordinate values of the
        ``"dimension"`` from ``flattened``.
        Values are tuples of variable name and the selector dict.
    """
    draws = {}
    selectors: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for name, var in dataset.items():
        stacked_dims = set(var.dims) - {skipdim}
        if not stacked_dims:
            # Scalar variable
            ds = var.values
            draws[name] = ds
            selectors[name] = (name, {})
        else:
            coordinates = [
                var[dname].values
                for dname in stacked_dims
            ]
            for cvals in itertools.product(*coordinates):
                sel = {
                    dn : cv
                    for dn, cv in zip(stacked_dims, cvals)
                }
                vals = var.sel(sel)
                assert vals.ndim == 1, sel
                cname = ", ".join(map(str, cvals))
                key = f"{name}[{cname}]"
                draws[key] = vals
                selectors[key] = (name, sel)

    flattened = xarray.DataArray(
        data=list(draws.values()),
        dims=("dimension", skipdim),
        coords={
            "dimension": list(draws.keys()),
        }
    ).T
    return flattened, selectors


def top_correlations(df_samples: pandas.DataFrame) -> pandas.DataFrame:
    """Find most correlating pairs of dimensions.

    Adapted from https://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html

    Parameters
    ----------
    df_samples
        A dataframe where variables are in columns and samples in rows.

    Returns
    -------
    corr
        A dataframe with columns ["first", "second", "correlation"]
    """
    df_samples = df_samples.copy()
    corr = df_samples.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    corr = corr * numpy.tri(*corr.values.shape, k=-1).T
    corr.index.name = "first"
    corr.columns.name = "second"
    # find the top n correlations
    corr = corr.stack()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).reset_index()
    # Rename the columns to something nice
    corr.columns = [*corr.columns[:2], "correlation"]
    return corr


def do_pca(
    pmodel,
    samples: xarray.Dataset,
) -> Tuple[
    sklearn.decomposition.PCA,
    pandas.DataFrame,
    Dict[str, Tuple[str, Dict[str, Any]]]
]:
    df_samples = samples.to_series().unstack()
    X = sklearn.preprocessing.StandardScaler().fit_transform(df_samples)
    pca = sklearn.decomposition.PCA().fit(X)
    return pca, df_samples


def pca_feature_weights(pca) -> numpy.ndarray:
    n_relevant = sum(numpy.cumsum(pca.explained_variance_ratio_) < 0.5)
    _log.info(f"The first {n_relevant} principal components explain ~50 % of the variance.")
    feature_weights = abs(pca.components_[:n_relevant]).sum(axis=0)
    return feature_weights
