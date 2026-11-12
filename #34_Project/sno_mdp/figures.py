"""Report figures from saved artifacts; no influence on policy decisions."""

import json
import os
import tempfile

import numpy as np

from .artifacts import read_json


def plot_overview(run):
    maps = json.loads((run / "map.json").read_text())
    frames = read_json(run / "trajectory.json.gz")
    z = np.array(maps["altitude"])
    reward = np.array(maps["reward"])
    x, y = np.indices(z.shape, dtype=float) * maps["spacing"]
    positions = np.array([f["position"] for f in frames])
    height = 0.8 * (z - z.min()) / (np.ptp(z) or 1)
    previous = {k: os.environ.get(k) for k in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    with tempfile.TemporaryDirectory(prefix="snomdp-figure-") as cache:
        for k, v in previous.items():
            if v is None:
                os.environ[k] = cache
        try:
            import matplotlib as mpl
            from matplotlib.figure import Figure
            from matplotlib.colors import LinearSegmentedColormap

            with mpl.rc_context(
                {"svg.hashsalt": "snomdp-report", "font.family": "DejaVu Sans"}
            ):
                fig = Figure(figsize=(10, 7), layout="constrained")
                ax = fig.add_subplot(111, projection="3d")
                colors = LinearSegmentedColormap.from_list(
                    "reward", ["#2455cf", "#3ea851", "#f4d94a"]
                )
                width = maps["spacing"] * 0.9
                ax.bar3d(
                    x.ravel(),
                    y.ravel(),
                    np.zeros(z.size),
                    width,
                    width,
                    height.ravel() + 1e-6,
                    color=colors((reward / (reward.max() or 1)).ravel()),
                    shade=True,
                    alpha=0.9,
                )
                px, py = positions[:, 0], positions[:, 1]
                ax.plot(
                    x[px, py] + width / 2,
                    y[px, py] + width / 2,
                    height[px, py] + 0.08,
                    color="#aa1212",
                    linewidth=2,
                )
                ax.set(
                    xlabel="x coordinate",
                    ylabel="y coordinate",
                    zlabel="Normalized altitude (display only)",
                    title="GP-Safety-Gym benchmark fields and simulated path\nReward colors: blue (low), green, yellow (high)",
                )
                fig.savefig(run / "overview.png", dpi=150)
                fig.savefig(run / "overview.svg", metadata={"Date": None})
        finally:
            for k, v in previous.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
