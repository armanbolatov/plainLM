"""Shared plot style — minimal, borderless, light blue-gray background.

Modelled on the figure style used in Schaipp/Gower/Taylor's papers: no black
spines, soft gray-blue axes background, white gridlines, restrained fonts.
Every plot script in this repo calls `apply_style()` before drawing.
"""
import matplotlib
import seaborn as sns

LW      = 2.2   # plot line width
S_LINE  = 34    # scatter marker size for line dots

# Soft, high-contrast-on-gray palette. Order matters: it is used as the
# categorical color cycle and indexed by the α-gradient helper below.
PALETTE = {
    'scion':           '#0173b2',
    'lngn_scion':      '#de8f05',
    'gngn_alpha_7e-2': '#029e73',
    'adamw':           '#d55e00',
    'muonmax_momo':    '#cc78bc',
    'ngnmdv1':         '#ca9161',
    'gngn_alpha_3e-2': '#fbafe4',
    'gngn_alpha_5e-2': '#56b4e9',
    'sfplus':          '#949494',
    'sfplus_wd':       '#949494',
}


def apply_style():
    sns.set_theme(
        style='darkgrid',
        context='notebook',
        rc={
            # Light blue-gray panel, white grid, no hard borders.
            'axes.facecolor':    '#EAEAF2',
            'figure.facecolor':  'white',
            'axes.edgecolor':    'white',
            'axes.linewidth':    0.0,
            'grid.color':        'white',
            'grid.linewidth':    1.0,
            'axes.grid':         True,

            # Ticks live inside the panel; no protruding marks.
            'xtick.bottom':      False,
            'ytick.left':        False,
            'xtick.color':       '#4A4A4A',
            'ytick.color':       '#4A4A4A',
            'axes.labelcolor':   '#2A2A2A',
            'text.color':        '#2A2A2A',

            'legend.frameon':    False,
            'lines.linewidth':   LW,
            'lines.solid_capstyle': 'round',

            # Vector-friendly text in the PDF output.
            'pdf.fonttype':      42,
            'ps.fonttype':       42,
            'savefig.bbox':      'tight',
            'savefig.pad_inches': 0.05,
        },
    )
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
        color=list(PALETTE.values())
    )


def alpha_colors(n, cmap='viridis'):
    """n colors sampled across a sequential map — for ordered α values."""
    cm = matplotlib.colormaps[cmap]
    return [cm(i / max(n - 1, 1)) for i in range(n)]


def shared_legend(fig, ax, ncols=None):
    """One legend under the whole figure, sourced from `ax`.

    Panels in these figures all draw the same series, so an in-axes legend is
    both redundant and guaranteed to occlude data somewhere. Hoisting it to the
    figure keeps placement deterministic no matter how the curves move.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels,
               loc='outside lower center',
               ncols=ncols or len(handles),
               frameon=False)


def minor_grid(ax, axis='both'):
    """Add faint white minor gridlines.

    A log-scaled axis only gets major ticks at each decade, which leaves large
    unbroken bands; the minor grid restores a readable reference density.
    """
    ax.minorticks_on()
    ax.grid(True, which='minor', axis=axis,
            color='white', linewidth=0.5, alpha=0.6)
    ax.grid(True, which='major', axis='both',
            color='white', linewidth=1.0)
    ax.tick_params(which='minor', length=0)
