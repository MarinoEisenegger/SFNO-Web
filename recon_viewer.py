"""
Reconstruction Viewer
---------------------
Run with:  streamlit run recon_viewer.py
Expects:   reconstructions/{variable}/{experiment_name}.pt
           where each .pt holds a 2-D array (Lat x Lon).
"""

import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
RECON_ROOT = Path("reconstructions")
MAX_COLS = 3

st.set_page_config(
    page_title="Reconstruction Viewer",
    layout="wide",
    page_icon="🌍",
)

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_tensor(path: str) -> np.ndarray:
    """Load a .pt file that may contain a torch.Tensor or a numpy array."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, np.ndarray):
        return obj
    raise ValueError(f"Unsupported type in {path}: {type(obj)}")


def scan_reconstructions(root: Path) -> dict[str, list[str]]:
    """Return {variable: [experiment_name, ...]} sorted alphabetically."""
    result = {}
    if not root.exists():
        return result
    for var_dir in sorted(root.iterdir()):
        if var_dir.is_dir():
            experiments = sorted(
                p.stem for p in var_dir.glob("*.pt")
            )
            if experiments:
                result[var_dir.name] = experiments
    return result


def render_panel(ax_top, ax_res, img: np.ndarray, label: str,
                 vmin: float, vmax: float, fig):
    """Draw a single reconstruction column (image + residual placeholder)."""
    ax_top.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax_top.set_title(label, fontsize=10, fontweight="bold", pad=6)
    ax_top.axis("off")

    # Residual row is filled in separately once the reference is known.
    ax_res._img_data = img  # stash for later


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 Recon Viewer")
    st.caption("Select a variable, then up to 3 experiments to compare.")

    catalog = scan_reconstructions(RECON_ROOT)

    if not catalog:
        st.error(
            f"No reconstructions found under **{RECON_ROOT.resolve()}**.\n\n"
            "Make sure the directory exists and contains `variable/experiment.pt` files."
        )
        st.stop()

    # Variable selector
    variable = st.selectbox(
        "🗂 Variable",
        options=list(catalog.keys()),
        help="Top-level folder name inside reconstructions/",
    )

    experiments = catalog[variable]

    st.markdown("---")
    st.markdown("**Experiments to compare**")

    selections: list[str | None] = []
    for i in range(MAX_COLS):
        opts = ["— (none) —"] + experiments
        choice = st.selectbox(
            f"Slot {i + 1}",
            options=opts,
            index=0 if i > 0 else 1,  # auto-select first experiment in slot 1
            key=f"exp_{i}",
        )
        selections.append(None if choice == "— (none) —" else choice)

    active = [(i, name) for i, name in enumerate(selections) if name]

    st.markdown("---")
    show_residual = st.checkbox("Show residual row", value=True)
    ref_idx = None
    if show_residual and len(active) > 1:
        ref_options = [name for _, name in active]
        ref_choice = st.selectbox(
            "Residual reference",
            options=ref_options,
            index=0,
            help="Residual = this experiment − others",
        )
        ref_idx = next(i for i, name in active if name == ref_choice)

    cmap_img = st.selectbox(
        "Colormap (images)", ["viridis", "plasma", "RdBu_r", "coolwarm", "inferno"], index=0
    )
    cmap_res = st.selectbox(
        "Colormap (residuals)", ["seismic", "RdBu_r", "bwr", "coolwarm"], index=0
    )

# ── Main area ────────────────────────────────────────────────────────────────
st.markdown(f"## `{variable}` — {len(active)} experiment(s) selected")

if not active:
    st.info("Select at least one experiment in the sidebar.")
    st.stop()

# Load data
arrays: dict[str, np.ndarray] = {}
for _, name in active:
    pt_path = RECON_ROOT / variable / f"{name}.pt"
    try:
        arrays[name] = load_tensor(str(pt_path))
    except Exception as e:
        st.error(f"Failed to load **{name}**: {e}")
        st.stop()

# Shared colour limits (union of all loaded arrays)
all_vals = np.concatenate([a.ravel() for a in arrays.values()])
global_vmin, global_vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

with st.sidebar:
    st.markdown("---")
    st.markdown("**Colour range**")
    use_custom = st.checkbox("Custom vmin / vmax", value=False)
    if use_custom:
        vmin = st.number_input("vmin", value=float(f"{global_vmin:.4g}"))
        vmax = st.number_input("vmax", value=float(f"{global_vmax:.4g}"))
    else:
        vmin, vmax = global_vmin, global_vmax

# ── Figure ───────────────────────────────────────────────────────────────────
n_cols = len(active)
n_rows = 2 if (show_residual and ref_idx is not None) else 1
row_labels = ["Reconstruction", "Residual"] if n_rows == 2 else ["Reconstruction"]

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5.5 * n_cols, 4.5 * n_rows),
    squeeze=False,
)
fig.patch.set_facecolor("#0e1117")

ref_img = arrays.get(selections[ref_idx]) if ref_idx is not None else None

for col, (slot_i, name) in enumerate(active):
    img = arrays[name]

    # Row 0 — reconstruction
    ax = axes[0, col]
    im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax,
                   cmap=cmap_img, interpolation="nearest")
    ax.set_title(name, color="white", fontsize=9, fontweight="bold", pad=5)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, fraction=0.046)
    cb.ax.tick_params(colors="white", labelsize=7)
    cb.set_label(variable, color="white", fontsize=7)

    # Row 1 — residual
    if n_rows == 2 and ref_img is not None:
        ax_r = axes[1, col]
        if name == selections[ref_idx]:
            ax_r.text(0.5, 0.5, "(reference)", ha="center", va="center",
                      color="grey", transform=ax_r.transAxes, fontsize=10)
            ax_r.set_facecolor("#0e1117")
            ax_r.axis("off")
        else:
            residual = ref_img - img
            abs_max = float(np.nanmax(np.abs(residual)))
            im_r = ax_r.imshow(residual, origin="lower",
                               vmin=-abs_max, vmax=abs_max,
                               cmap=cmap_res, interpolation="nearest")
            ax_r.set_title(f"{selections[ref_idx]} − {name}",
                           color="white", fontsize=8, pad=5)
            ax_r.axis("off")
            cb_r = fig.colorbar(im_r, ax=ax_r, orientation="horizontal",
                                pad=0.02, fraction=0.046)
            cb_r.ax.tick_params(colors="white", labelsize=7)
            cb_r.set_label("Δ " + variable, color="white", fontsize=7)

# Row labels on the left
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, color="white", fontsize=10, labelpad=8)
    axes[row, 0].yaxis.label.set_visible(True)

fig.tight_layout(pad=1.2)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Stats table ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Quick stats")

rows = []
for _, name in active:
    a = arrays[name]
    row = {
        "Experiment": name,
        "Min": f"{np.nanmin(a):.4f}",
        "Max": f"{np.nanmax(a):.4f}",
        "Mean": f"{np.nanmean(a):.4f}",
        "Std": f"{np.nanstd(a):.4f}",
    }
    if ref_img is not None and name != selections[ref_idx]:
        res = ref_img - a
        row["RMSE vs ref"] = f"{np.sqrt(np.nanmean(res**2)):.4f}"
        row["MAE vs ref"] = f"{np.nanmean(np.abs(res)):.4f}"
    rows.append(row)

st.dataframe(rows, use_container_width=True, hide_index=True)