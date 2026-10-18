import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import Normalize
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@torch.no_grad()
def plot_sample(
    reference_theta: torch.Tensor,
    diffusion_theta_mean: torch.Tensor,
    diffusion_theta_gamma_mean: torch.Tensor,
    is_theta_mean: torch.Tensor,
    filename: str,
) -> None:
    diffusion_theta_error = (diffusion_theta_mean - reference_theta).abs()
    diffusion_theta_gamma_error = (diffusion_theta_gamma_mean - reference_theta).abs()
    is_theta_error = (is_theta_mean - reference_theta).abs()

    def _to_numpy(t: torch.Tensor):
        return t.detach().cpu().squeeze().numpy()

    ref = _to_numpy(reference_theta)
    diffusion_mean = _to_numpy(diffusion_theta_mean)
    diffusion_gamma_mean = _to_numpy(diffusion_theta_gamma_mean)
    is_mean = _to_numpy(is_theta_mean)
    diffusion_error = _to_numpy(diffusion_theta_error)
    diffusion_gamma_error = _to_numpy(diffusion_theta_gamma_error)
    is_error = _to_numpy(is_theta_error)

    common = dict(
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="nearest",
        aspect="equal",
    )

    # Shared parameter scale
    vmin = min(ref.min(), diffusion_mean.min(), diffusion_gamma_mean.min(), is_mean.min())
    vmax = max(ref.max(), diffusion_mean.max(), diffusion_gamma_mean.max(), is_mean.max())

    # Shared scale for nonnegative absolute errors.
    error_vmax = max(diffusion_error.max(), diffusion_gamma_error.max(), is_error.max())

    fig, axes = plt.subplots(
        2, 4, figsize=(20, 10), constrained_layout=True
    )
    ax = axes.ravel()

    try:
        im_ref = ax[0].imshow(ref.T, cmap="RdBu_r", vmin=vmin, vmax=vmax, **common)
        ax[1].imshow(diffusion_mean.T, cmap="RdBu_r", vmin=vmin, vmax=vmax, **common)
        ax[2].imshow(diffusion_gamma_mean.T, cmap="RdBu_r", vmin=vmin, vmax=vmax, **common)
        ax[3].imshow(is_mean.T, cmap="RdBu_r", vmin=vmin, vmax=vmax, **common)
        ax[4].axis("off")

        im_err = ax[5].imshow(diffusion_error.T, cmap="magma", vmin=0.0, vmax=error_vmax, **common)
        ax[6].imshow(diffusion_gamma_error.T, cmap="magma", vmin=0.0, vmax=error_vmax, **common)
        ax[7].imshow(is_error.T, cmap="magma", vmin=0.0, vmax=error_vmax, **common)

        titles = [
            "Reference parameter",
            "Estimated parameter (diffusion with $A=I$)",
            "Estimated parameter (diffusion with $A=\gamma I-\Delta$)",
            "Estimated parameter (IS)",
            "",
            "Absolute error (diffusion with $A=I$)",
            "Absolute error (diffusion with $A=\gamma I-\Delta$)",
            "Absolute error (IS)",
        ]
        for i in (0, 1, 2, 3, 5, 6, 7):
            ax[i].set_title(titles[i])

        fig.colorbar(im_ref, ax=list(ax[:4]), shrink=0.9)
        fig.colorbar(im_err, ax=list(ax[5:]), shrink=0.9)

        fig.savefig(filename, bbox_inches="tight")
    finally:
        plt.close(fig)


@torch.no_grad()
def plot_sample_3d(
    reference_theta: torch.Tensor,
    estimated_theta_mean: torch.Tensor,
    estimated_theta_error: torch.Tensor,
    estimated_theta_std: torch.Tensor,
    filename: str = "./result_3d.pdf",
    cmap: str = "twilight",
    surface_points: int = 35,
    show_projections: bool = True,
) -> None:
    """Four faceted surfaces with gray maximum profiles on the back walls.

    Each tensor must squeeze to the same (nx, ny) shape on [0, 1]^2.
    Reference and mean share color and height scales. Error and standard
    deviation have independent scales. Vertical limits extend slightly beyond
    the data range, and wall profiles close at the lower vertical limit.
    """
    fields = [t.detach().cpu().squeeze().numpy() for t in (
        reference_theta, estimated_theta_mean,
        estimated_theta_error, estimated_theta_std,
    )]
    if any(z.ndim != 2 or min(z.shape) < 2 for z in fields):
        raise ValueError("Each tensor must squeeze to a 2D grid of at least 2 x 2.")
    if any(z.shape != fields[0].shape for z in fields):
        raise ValueError("All four grids must have the same shape.")
    if any(not np.isfinite(z).all() for z in fields):
        raise ValueError("The grids must contain only finite values.")
    if not isinstance(surface_points, int) or surface_points < 2:
        raise ValueError("surface_points must be an integer of at least 2.")

    ref, mean, err, std = fields
    nx, ny = ref.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    # indexing='ij' preserves the orientation of your original imshow(ref.T).
    X, Y = np.meshgrid(x, y, indexing="ij")
    shared = (float(min(ref.min(), mean.min())),
              float(max(ref.max(), mean.max())))
    limits = [shared, shared] + [
        (float(z.min()), float(z.max()))
        for z in (err, std)
    ]
    titles = ["Reference parameter", "Estimated parameter (mean)", "Error", "Std"]
    zlabels = [r"$\theta$", r"$\hat{\theta}$", "Error", r"$\sigma$"]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 13,
                         "mathtext.fontset": "dejavusans",
                         "axes.edgecolor": "0.6", "axes.linewidth": 0.7}):
        fig, axes = plt.subplots(
            1, 4, figsize=(24, 6.5),
            subplot_kw={"projection": "3d", "computed_zorder": False},
        )
        try:
            for ax, Z, (lo, hi), title, zlabel in zip(
                axes, fields, limits, titles, zlabels
            ):
                # Start just below the minimum, with 5% padding.
                span = hi - lo
                if span == 0:
                    span = max(abs(lo), 1e-6)
                bottom = lo - 0.05 * span
                top = hi + 0.08 * span
                if lo == hi:
                    pad = 0.5 * max(abs(lo), 1.0)
                    norm = Normalize(lo - pad, hi + pad)
                else:
                    norm = Normalize(lo, hi)

                if show_projections:
                    # Maximum profiles, not probability marginals.
                    x_profile, y_profile = Z.max(axis=1), Z.max(axis=0)
                    walls = [
                        [(x[0], y[-1], bottom)]
                        + list(zip(x, np.full(nx, y[-1]), x_profile))
                        + [(x[-1], y[-1], bottom)],
                        [(x[0], y[0], bottom)]
                        + list(zip(np.full(ny, x[0]), y, y_profile))
                        + [(x[0], y[-1], bottom)],
                    ]
                    ax.add_collection3d(Poly3DCollection(
                        walls, facecolors="0.75", edgecolors="none",
                        alpha=0.45, zorder=1,
                    ))

                ax.plot_surface(
                    X, Y, Z, cmap=cmap, norm=norm,
                    rcount=min(surface_points, nx),
                    ccount=min(surface_points, ny),
                    linewidth=0, edgecolor="none", antialiased=True,
                    shade=False, zorder=2,
                )
                ax.view_init(elev=28, azim=-58)
                ax.set_box_aspect((1, 1, 0.65))
                ax.set(xlim=(0, 1), ylim=(0, 1),
                       zlim=(bottom, top))
                ax.set_xticks([0.2, 0.8])
                ax.set_yticks([0.2, 0.8])
                ax.set_zticks(np.linspace(lo, hi, 3) if hi > lo else [lo])
                ax.zaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
                ax.set_xlabel(r"$x_1$", fontsize=23, labelpad=10)
                ax.set_ylabel(r"$x_2$", fontsize=23, labelpad=10)
                ax.set_zlabel(zlabel, fontsize=21, labelpad=10)
                ax.tick_params(labelsize=13, pad=2)
                ax.set_title(title, fontsize=16, pad=15, bbox={
                    "boxstyle": "round,pad=0.3", "facecolor": "white",
                    "edgecolor": "0.55", "linewidth": 1.2,
                })
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.set_pane_color((0.93, 0.93, 0.93, 1.0))
                    # Matplotlib's 3D grid styling uses this internal dictionary.
                    axis._axinfo["grid"].update(color="white", linewidth=1.3)
                ax.grid(True)

            fig.subplots_adjust(left=0.01, right=0.97, bottom=0.10,
                                top=0.86, wspace=0.15)
            fig.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)

