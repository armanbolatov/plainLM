"""Shared plot style. Seaborn whitegrid + default fonts + thick lines.

Kept minimal on purpose: only change what actually helps readability
(line width, marker size, tight layout). Everything else — font family,
font sizes, tick style, spines — is the matplotlib/seaborn default.
"""
import matplotlib
import seaborn as sns

LW         = 2.5   # plot line width
S_LINE     = 60    # scatter marker size for line dots


def apply_style():
    sns.set_theme(style='whitegrid', context='notebook')
    r = matplotlib.rcParams
    r['pdf.fonttype']    = 42
    r['ps.fonttype']     = 42
    r['savefig.bbox']    = 'tight'
    r['savefig.pad_inches'] = 0.05


PALETTE = {
    'scion':           '#1f77b4',
    'lngn_scion':      '#ff7f0e',
    'gngn_alpha_7e-2': '#2ca02c',
    'adamw':           '#d62728',
    'muonmax_momo':    '#9467bd',
    'ngnmdv1':         '#8c564b',
    'gngn_alpha_3e-2': '#e377c2',
    'gngn_alpha_5e-2': '#17becf',
}
