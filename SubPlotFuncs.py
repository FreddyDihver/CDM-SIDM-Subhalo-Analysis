from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

##############################
# Subhalo plotting functions #
##############################
def text_box(ax, text, x=0.3, y=0.98, fontsize=14, color='k'):
    """Creates a text box on the given axes with specified text and formatting.

    Args:
        ax (Axes): ax to plot the text box on
        text (str): text to display in the box
        x (float, optional): x-coordinate of the text box. Defaults to 0.3.
        y (float, optional): y-coordinate of the text box. Defaults to 0.98.
        fontsize (int, optional): Font size of the text. Defaults to 14.
        color (str, optional): Color of the text. Defaults to 'k'.
    """
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, verticalalignment='top', bbox=props, color=color)


def save_figure(fig, filepath):
    """Saves the figure to the specified filepath, creating directories if they do not exist.
    Args:
        fig (figure): figure to save
        filepath (str): where to save the figure
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)


def plot_subhalo(coord, CM, MaxPot, r200, clean, c1=0, c2=1):
    """Plot the particle distribution of a subhalo with optional markers for center of mass and potential minimum, 
    and a circle for the virial radius.

    Args:
        coord (ndarray): Array of particle coordinates
        CM (ndarray): Center of mass position of the subhalo
        MaxPot (ndarray): maximum potential position of the subhalo
        r200 (float): virial radius of the subhalo
        clean (bool): if True, only plot the particle distribution without markers. Defaults to False.
        c1 (int, optional): vertical axis (0,1,2). Defaults to 0.
        c2 (int, optional): horizontal axis (0,1,2). Defaults to 1.
    """
    fig = plt.figure(figsize=(6.9, 6.9))
    plt.style.use('dark_background')
    ax = plt.axes([0.1, 0.1, 0.87, 0.87])

    ax.scatter(coord[:, c1], coord[:, c2], rasterized=True, s=0.5, c='w')

    # Add markers for center of mass and potential minimum, and a circle for the virial radius if not in clean mode
    if not clean:
        ax.scatter(CM[c1], CM[c2], marker='x', c='r')
        ax.scatter(MaxPot[c1], MaxPot[c2], marker='x', c='b')
        try:
            theta = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(r200 * np.cos(theta) + MaxPot[c1],
                    r200 * np.sin(theta) + MaxPot[c2])
        except TypeError:
            pass

    # Set labels, ticks, and aspect ratio for the plot
    ax.set_xlabel('x-coordinate (kpc)', fontsize=16)
    ax.set_ylabel('y-coordinate (kpc)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.tick_params(axis='x', labelrotation=45)
    plt.axis('equal')

    # Return the figure and axes for further plotting if needed
    return fig, ax

def info_text(num_particles, group_id, m200, virratio, relax):
    """Generates a string containing information about the subhalo for display in a text box on the particle plot.

    Args:
        num_particles (int): The number of dark matter particles in the subhalo
        group_id (int): The group ID of the subhalo
        m200 (float): The mass of the subhalo within R200
        virratio (float): The virial ratio of the subhalo
        relax (float): The relaxation parameter of the subhalo

    Returns:
        str: A string containing the subhalo information
    """
    return (
        f'{num_particles} particles\n'
        f'Group {group_id}\n'
        f'M200 = {m200:.2E} Msol\n'
        f'Virial ratio = {virratio:.4f}\n'
        f'Relaxation (rel to R200): {relax:.4f}'
    )

def colorbar_labels(ticks)->list:
    """Create formatted colorbar labels for surface density.

    Args: 
        ticks (array-like): Array of tick values for the colorbar.

    Returns:
        list: A list of formatted label strings.
    """
    labels = []

    for i, tick in enumerate(ticks):
        exponent = int(np.log10(tick))
        coefficient = round(tick / 10**exponent, 1)

        if i == 0:
            labels.append(
                fr'$\leq {coefficient}\cdot 10^{{{exponent}}} '
                r'M_\odot/kpc^2$'
            )
        else:
            labels.append(fr'$10^{{{exponent}}} M_\odot/kpc^2$')

    return labels


def plot_subhalo_colormesh(coord, CM, MaxPot, pmass, r200, clean, c1=0, c2=1)->tuple:
    """Plot a subhalo as a projected surface-density colormesh.
    Args:
        coord (ndarray): Array of particle coordinates
        CM (ndarray): Center of mass position of the subhalo
        MaxPot (ndarray): Position of the particle with the extremum potential in the subhalo
        pmass (float): Particle mass
        r200 (float): Virial radius of the subhalo
        clean (bool): If True, only plot the particle distribution without markers. Defaults to False.
        c1 (int, optional): Vertical axis (0,1,2). Defaults to 0.
        c2 (int, optional): Horizontal axis (0,1,2). Defaults to 1.

    Returns:
        tuple: A tuple containing the figure and axes for further plotting if needed.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6.9))
    ax = plt.axes([0.1, 0.1, 0.87, 0.87])

    nbins = max(200, int(len(coord) // 1e4))
    count_grid, xedges, yedges = np.histogram2d(coord[:, c1], coord[:, c2], nbins)

    dx = (xedges[-1] - xedges[0]) / nbins
    dy = (yedges[-1] - yedges[0]) / nbins
    cellarea = dx * dy
    weights = np.full(len(coord), pmass / cellarea)

    density_grid, _, _, im = ax.hist2d(
        coord[:, c1],
        coord[:, c2],
        nbins,
        weights=weights,
        cmap=plt.cm.CMRmap,
        cmin=1,
        norm=LogNorm(),
    )

    finite_density = density_grid[np.isfinite(density_grid) & (density_grid > 0)]
    if len(finite_density) > 0:
        ticks = np.append(finite_density.min(), np.logspace(7, 12, num=6))
        cbar = fig.colorbar(im, ax=ax, ticks=ticks)
        cbar.ax.set_yticklabels(colorbar_labels(ticks), fontsize=14)

    if not clean:
        ax.scatter(CM[c1], CM[c2], marker='x', c='r')
        ax.scatter(MaxPot[c1], MaxPot[c2], marker='x', c='b')

        try:
            theta = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(r200 * np.cos(theta) + MaxPot[c1],
                    r200 * np.sin(theta) + MaxPot[c2])
        except TypeError:
            pass

    ax.set_xlabel('x-coordinate (kpc)', fontsize=16)
    ax.set_ylabel('y-coordinate (kpc)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.tick_params(axis='x', labelrotation=45)
    plt.axis('equal')

    return fig, ax