import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def create_figs(results):
    #Create figs
    figs = []
    for i,r in enumerate(results):
        figs_ = []
        for j,rr in enumerate(r):
            smile = rr["smile"]
            weights = rr["weights"]
            f = draw_chem_activations(smile=smile, weights=weights, colorbar=False,
                                      radius=0.03, colormap="Blues", 
                                      vmax=1.0, vmin=0.0)
            figs_.append(f)
        figs.append(figs_)
    return figs


def create_im_arrs(figs):
    #Create image arrays from figures
    ims_arr = []
    for i,f in enumerate(figs):
        ims_arr_ = []
        for j,ff in enumerate(f):
            X = np.array(ff.canvas.renderer._renderer)
            ims_arr_.append(X)
        ims_arr.append(ims_arr_)
    return ims_arr


def draw_atom_disks(mol, weights, radius=0.05, step=0.001):
    """
    Draw disks of fixed radius around each atom's coordinate.
    `weights` controls the color of each disk.
    """
    x = np.arange(0, 1, step)
    y = np.arange(0, 1, step)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i, (c_x, c_y) in mol._atomPs.items():
        base = (X - c_x) ** 2 + (Y - c_y) ** 2
        circle_mask = (base < radius**2)
        circle = circle_mask.astype('float') * weights[i]
        Z += circle
    return X, Y, Z


def draw_chem_activations(smile, weights, colorbar=False, colormap = 'RdBu',
                          step=0.001, size=(250, 250), radius=0.025,
                          coord_scale=1.5, title=None, vmax=1.0, vmin=-1.0):
    """
    Draw scalar activations on each
    """
    mol = Chem.MolFromSmiles(smile)
    cmap = plt.cm.get_cmap(colormap)
    fig = Draw.MolToMPL(mol, coordScale=coord_scale, size=size, **{}, )
    ax = fig.axes[0]
    x, y, z = draw_atom_disks(mol, radius=radius, weights=weights, step=step)
    ax.imshow(z, cmap=cmap, interpolation='bilinear', origin='lower',
                  extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax)

    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    if colorbar:
        plt.colorbar(sm, fraction=0.035, pad=0.04)
    if title:
        plt.title(title)

    return fig


def plot_image_grid(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    super_col_labels=None,
                    file_name=None,
                    dpi=224,
                    c=10,
                    fontsize=24,
                    col_rotation=22.5):
    """
    Forked from https://github.com/albermax/innvestigate/blob/master/examples/utils.py
    """
    n_rows = len(grid)
    n_cols = len(grid[0])

    plt.clf()
    plt.rc("font", family="sans-serif", size=fontsize)

    f = plt.figure(figsize = (c*n_cols, c*n_rows))
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows, n_cols], loc=[r,c])
            ax.imshow(grid[r][c], interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])

            if not r: #column labels
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=col_rotation,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

                if super_col_labels != [] and c % 2 == 0:
                    label = super_col_labels[c // 2]
                    x_adjust = len(label) / 100
                    ax.text(x = 1 - x_adjust,
                            y = 1.2,
                            s = label,
                            transform=ax.transAxes,
                            fontdict={"fontsize": 50, "weight": 10})
            if not c: #row labels
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                    ax.set_ylabel(''.join(txt_left),
                                  rotation=0,
                                  verticalalignment='center',
                                  horizontalalignment='right',
                                  )
            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(''.join(txt_right),
                                  rotation=0,
                                  verticalalignment='center',
                                  horizontalalignment='left'
                                   )
    if not file_name:
        plt.show()
    else:
        print ('saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi)
