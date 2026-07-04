"""Plotting helpers built on matplotlib."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_quiver(X, Y, u, v, path, title="Velocity"):
    fig, ax = plt.subplots(figsize=(5, 5))
    step = max(1, X.shape[0] // 24)
    ax.quiver(X[::step, ::step], Y[::step, ::step], u[::step, ::step], v[::step, ::step])
    ax.set_title(title); ax.set_aspect("equal")
    _save(fig, path)


def plot_scalar(field, path, title=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field.T, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_title(title)
    _save(fig, path)


def plot_error_heatmap(u, v, u_ex, v_ex, path, title="|u-u_exact|"):
    err = np.sqrt((u - u_ex) ** 2 + (v - v_ex) ** 2)
    plot_scalar(err, path, title=title)


def plot_kinetic_energy(t_arr, ke_num, ke_exact, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(t_arr, ke_num, label="numerical")
    ax.plot(t_arr, ke_exact, "--", label="exact")
    ax.set_xlabel("t"); ax.set_ylabel("kinetic energy"); ax.legend()
    _save(fig, path)


def plot_metric_vs_re(rows, key_y: str, path, ylabel: Optional[str] = None, log: bool = False):
    fig, ax = plt.subplots(figsize=(5, 4))
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        sub.sort(key=lambda r: r["Re"])
        xs = [r["Re"] for r in sub]
        ys = [r[key_y] for r in sub]
        ax.plot(xs, ys, "-o", label=m)
    ax.set_xlabel("Re"); ax.set_ylabel(ylabel or key_y)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_compression_vs_error(rows, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in rows:
        ax.scatter(r["compression_ratio"], r["l2_velocity_error"], label=r["method"])
    ax.set_xlabel("compression ratio"); ax.set_ylabel("L2 velocity error")
    _save(fig, path)


def plot_associator_field(A, path, title="Associator field"):
    plot_scalar(A, path, title=title)


def plot_associator_vs_error(A, u, v, u_ex, v_ex, path):
    err = np.sqrt((u - u_ex) ** 2 + (v - v_ex) ** 2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(A.ravel(), err.ravel(), s=2, alpha=0.4)
    ax.set_xlabel("associator norm"); ax.set_ylabel("local |u-u_exact|")
    _save(fig, path)
