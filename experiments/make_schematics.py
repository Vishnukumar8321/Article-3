"""Figures 1-4: the schematics.

Figure 1 is **not** a redrawing of the dataset paper's testbed diagram. It is
derived from the released flows themselves -- endpoint roles are inferred from
observed traffic (who serves port 1883, who scans many destinations, who only
ever receives) so that the figure describes the data actually used rather than
the testbed as described. The inference rules are printed into the figure's
companion JSON so a reader can check them.

Figures 2-4 are conceptual and drawn with the same theme tokens as the data
figures, so the plate reads as one system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import DATA_METADATA, RESULTS_FIGURES, get_logger, save_json  # noqa: E402
from src.visualization import LIGHT, Theme, apply_theme  # noqa: E402

log = get_logger("schematics")


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


def _box(ax, xy, w, h, label, theme, face=None, ink=None, fontsize=7,
         radius=0.06, lw=0.0, edge=None, weight="normal"):
    face = face if face is not None else theme.surface
    ink = ink if ink is not None else theme.ink_primary
    patch = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, facecolor=face,
        edgecolor=edge if edge is not None else theme.axis,
    )
    ax.add_patch(patch)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize,
            color=ink, zorder=5, fontweight=weight, linespacing=1.35)
    return patch


def _arrow(ax, a, b, theme, color=None, lw=1.1, style="-|>", rad=0.0,
           shrink=7.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=9,
        color=color or theme.ink_muted, linewidth=lw, linestyle=ls,
        shrinkA=shrink, shrinkB=shrink,
        connectionstyle=f"arc3,rad={rad}", zorder=2))


# --------------------------------------------------------------------------- #
# Figure 1 -- the testbed as the released flows describe it
# --------------------------------------------------------------------------- #


def infer_roles(representation: str = "uniflow") -> Dict[str, object]:
    """Assign an endpoint role from observed traffic, not from the paper.

    The rules are deliberately conservative and each one is checked against a
    quantity printed into the companion JSON, so a reader can audit every
    assignment. Anything the rules cannot place is labelled ``unclassified``
    rather than guessed.
    """
    from src.data_loader import load_representation

    df = load_representation(representation)
    y = df["multiclass_label"].to_numpy()
    src_str = df["ip_src"].astype(str)
    dst_str = df["ip_dst"].astype(str)

    roles: Dict[str, str] = {}
    evidence: Dict[str, Dict[str, object]] = {}
    endpoints = sorted(set(src_str) | set(dst_str))

    for ip in endpoints:
        as_src = (src_str == ip).to_numpy()
        as_dst = (dst_str == ip).to_numpy()
        n_src, n_dst = int(as_src.sum()), int(as_dst.sum())
        dst_peers = int(df.loc[as_src, "ip_dst"].nunique())
        rx_1883 = int((as_dst & (df["prt_dst"] == 1883).to_numpy()).sum())
        rx_22 = int((as_dst & (df["prt_dst"] == 22).to_numpy()).sum())
        attack_classes = sorted(set(y[as_src & (y > 0)].tolist()))

        if len(attack_classes) >= 3:
            role = "attacker"
        elif n_dst > 0 and rx_1883 / n_dst > 0.90:
            role = "MQTT broker"
        elif n_dst > 0 and rx_22 / n_dst > 0.20:
            role = "SSH / Sparta victim"
        elif n_src == 0:
            role = "external host"
        elif dst_peers >= 10:
            role = "gateway"
        elif n_src >= 10_000 and dst_peers <= 4:
            role = "MQTT sensor"
        elif n_src > 0:
            role = "scan-exposed host"
        else:
            role = "unclassified"

        roles[ip] = role
        evidence[ip] = {
            "flows_as_source": n_src, "flows_as_destination": n_dst,
            "distinct_destinations": dst_peers,
            "received_on_1883": rx_1883, "share_inbound_1883":
                round(rx_1883 / n_dst, 4) if n_dst else None,
            "received_on_22": rx_22, "share_inbound_22":
                round(rx_22 / n_dst, 4) if n_dst else None,
            "attack_classes_as_source": attack_classes,
            "assigned_role": roles[ip],
        }

    # Aggregate traffic between role groups, for the figure's arrows.
    role_of = {ip: roles[ip] for ip in endpoints}
    pair_counts: Dict[str, Dict[str, int]] = {}
    src_roles = src_str.map(role_of)
    dst_roles = dst_str.map(role_of)
    grouped = df.assign(_sr=src_roles, _dr=dst_roles, _y=y)
    for (sr, dr), sub in grouped.groupby(["_sr", "_dr"]):
        pair_counts.setdefault(sr, {})[dr] = int(len(sub))
    attack_paths = []
    for (sr, dr, cls), sub in grouped[grouped["_y"] > 0].groupby(["_sr", "_dr", "_y"]):
        attack_paths.append({"from_role": sr, "to_role": dr,
                             "class": int(cls), "flows": int(len(sub))})
    attack_paths.sort(key=lambda d: -d["flows"])

    out = {
        "representation": representation,
        "rules": {
            "attacker": "sources flows of >=3 distinct attack classes",
            "MQTT broker": ">90% of inbound flows arrive on port 1883",
            "SSH / Sparta victim": ">20% of inbound flows arrive on port 22",
            "external host": "never appears as a source",
            "gateway": ">=10 distinct destinations and not an attacker",
            "MQTT sensor": ">=10000 outbound flows to <=4 destinations",
            "scan-exposed host": "sources traffic but meets none of the above",
        },
        "note": ("An endpoint can source attack-labelled flows without being "
                 "the attacker: the dataset labels both directions of an "
                 "attack exchange, so the broker's replies to a brute-force "
                 "attempt carry is_attack=1. The attacker rule therefore keys "
                 "on the NUMBER OF DISTINCT attack classes sourced, which only "
                 "192.168.2.5 satisfies."),
        "roles": roles,
        "evidence": evidence,
        "counts": {r: sum(1 for v in roles.values() if v == r)
                   for r in sorted(set(roles.values()))},
        "role_to_role_flow_counts": pair_counts,
        "attack_paths_by_role": attack_paths[:20],
    }
    save_json(out, DATA_METADATA / f"endpoint_roles_{representation}.json")
    return out


def figure1_testbed(roles: Optional[Dict[str, object]] = None,
                    theme: Theme = LIGHT) -> Path:
    """Role-level view: one box per role, arrows aggregated over endpoints.

    Drawn at role level on purpose. An endpoint-level diagram of 24 nodes and
    106 ordered pairs is Figure 6, generated from the data; repeating it here
    with hand-placed arrows would only add crossings and a chance to be wrong.
    """
    roles = roles or infer_roles()
    r = roles["roles"]
    counts = roles["counts"]
    by_role: Dict[str, List[str]] = {}
    for ip, role in r.items():
        by_role.setdefault(role, []).append(ip)
    for v in by_role.values():
        v.sort()

    def members(role: str, limit: int = 4) -> str:
        ips = by_role.get(role, [])
        if not ips:
            return ""
        if len(ips) <= limit:
            return "\n".join(ips)
        return "\n".join(ips[:limit]) + f"\n+{len(ips) - limit} more"

    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.25, 6.2)
    ax.set_axis_off()

    spec = [
        ("MQTT sensor",         (1.55, 4.35), 2.4, 1.55, theme.categorical[0]),
        ("scan-exposed host",   (1.55, 1.30), 2.4, 1.30, theme.categorical[5]),
        ("MQTT broker",         (5.65, 4.35), 2.1, 1.00, theme.categorical[2]),
        ("gateway",             (5.65, 1.30), 2.1, 0.85, theme.ink_muted),
        ("external host",       (1.55, 2.85), 2.4, 0.62, theme.axis),
        ("SSH / Sparta victim", (9.85, 2.30), 2.2, 0.95, theme.categorical[4]),
        ("attacker",            (9.85, 4.85), 2.2, 0.95, theme.categorical[1]),
    ]
    pos: Dict[str, Tuple[float, float]] = {}
    for role, xy, w, h, colour in spec:
        n = counts.get(role, 0)
        if n == 0:
            continue
        label = f"{role}  (n={n})\n{members(role)}"
        _box(ax, xy, w, h, label, theme, face=colour, ink=theme.surface,
             fontsize=5.3, radius=0.10)
        pos[role] = xy

    def link(a, b, colour, lw, label=None, rad=0.0, ls="-", lx=None, ly=None,
             shrink=26):
        if a not in pos or b not in pos:
            return
        _arrow(ax, pos[a], pos[b], theme, color=colour, lw=lw, rad=rad, ls=ls,
               shrink=shrink)
        if label:
            mx = lx if lx is not None else (pos[a][0] + pos[b][0]) / 2
            my = ly if ly is not None else (pos[a][1] + pos[b][1]) / 2
            ax.text(mx, my, label, fontsize=5.4, color=colour, ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.14", fc=theme.surface,
                              ec="none"))

    link("MQTT sensor", "MQTT broker", theme.categorical[2], 1.6,
         "publish / subscribe\non 1883", ly=4.80)
    link("scan-exposed host", "gateway", theme.ink_muted, 0.9,
         "local traffic", ly=1.50)
    link("attacker", "MQTT broker", theme.categorical[1], 1.5,
         "MQTT brute-force", rad=0.12, lx=7.85, ly=5.30)
    link("attacker", "SSH / Sparta victim", theme.categorical[1], 1.5,
         "SSH brute-force", rad=-0.05, lx=11.30, ly=3.58)
    # The scan reaches the sensors; routed low so it clears the middle column.
    link("attacker", "MQTT sensor", theme.categorical[1], 1.5, None,
         rad=-0.40, ls=(0, (4, 2)))
    ax.text(5.65, 0.42, "aggressive scan / UDP scan", fontsize=5.4,
            color=theme.categorical[1], ha="center",
            bbox=dict(boxstyle="round,pad=0.16", fc=theme.surface, ec="none"))
    ax.text(1.55, 2.28, "receive-only", fontsize=5.0,
            color=theme.ink_muted, ha="center")

    ax.set_title("Endpoint roles and attack paths as observed in the released "
                 "uni-flow records", loc="left", fontsize=8.5, pad=6)
    ax.annotate("roles assigned by rules over observed traffic (ports served, "
                "scan fan-out, distinct attack classes sourced); rules and "
                "per-endpoint evidence in data/metadata/endpoint_roles_uniflow.json",
                (0.0, -0.05), xycoords="axes fraction", fontsize=5.2,
                color=theme.ink_muted)
    return _save(fig, RESULTS_FIGURES / "fig01_testbed_observed.png")


# --------------------------------------------------------------------------- #
# Figure 2 -- flow table to communication multigraph
# --------------------------------------------------------------------------- #


def figure2_flow_to_graph(theme: Theme = LIGHT) -> Path:
    """Left: four rows of the flow table. Right: the graph they become.

    Short node aliases are used so the labels fit inside the markers; the
    aliases are resolved in the row of the table they came from.
    """
    apply_theme(theme)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.1, 2.75),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in (axl, axr):
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.0, 6.2)

    alias = {"10.0.0.11": "S", "192.168.2.5": "A", "192.168.1.7": "B",
             "10.0.0.12": "S'"}
    header = ["ip_src", "ip_dst", "prt_dst", "num_pkts", "is_attack", "edge"]
    rows = [
        ["10.0.0.11",   "192.168.1.7", "1883",  "9",  "0", "z\u2081"],
        ["10.0.0.11",   "192.168.1.7", "1883",  "7",  "0", "z\u2082"],
        ["192.168.2.5", "192.168.1.7", "1883",  "4",  "1", "z\u2083"],
        ["192.168.2.5", "10.0.0.12",   "43112", "1",  "1", "z\u2084"],
    ]
    xs = [0.15, 2.15, 4.25, 5.70, 7.25, 8.85]
    axl.text(0.0, 5.6, "one row per flow record", fontsize=7.2,
             color=theme.ink_secondary, fontweight="bold")
    for x, h in zip(xs, header):
        axl.text(x, 4.95, h, fontsize=5.6, color=theme.ink_muted,
                 fontweight="bold")
    axl.plot([0.0, 9.7], [4.72, 4.72], color=theme.axis, lw=0.6)
    for i, row in enumerate(rows):
        yy = 4.15 - i * 0.78
        attack = row[4] == "1"
        ink = theme.categorical[1] if attack else theme.ink_primary
        for x, cell in zip(xs, row):
            axl.text(x, yy, cell, fontsize=5.6, color=ink)
        axl.plot([0.0, 9.7], [yy - 0.36, yy - 0.36], color=theme.grid, lw=0.4)
    legend = "  ".join(f"{a} = {ip}" for ip, a in alias.items())
    axl.text(0.0, 0.55, legend, fontsize=5.2, color=theme.ink_muted)
    axl.text(0.0, 0.12, "rows 1 and 2 share an endpoint pair but are separate "
                        "flows", fontsize=5.6, color=theme.ink_muted)

    # -- right: the multigraph ------------------------------------------- #
    pos = {"S": (1.7, 4.9), "A": (1.7, 1.7), "B": (7.6, 4.0), "S'": (7.9, 1.5)}
    colour = {"S": theme.categorical[0], "A": theme.categorical[1],
              "B": theme.categorical[2], "S'": theme.categorical[0]}
    # (rad, label, explicit label position) -- placed by hand because four
    # edges between four nodes leaves no room for an automatic rule.
    edges = [("S", "B", 0.30, theme.ink_muted, "z\u2081", (4.5, 5.05)),
             ("S", "B", 0.02, theme.ink_muted, "z\u2082", (4.9, 3.55)),
             ("A", "B", 0.26, theme.categorical[1], "z\u2083", (3.6, 2.15)),
             ("A", "S'", -0.20, theme.categorical[1], "z\u2084", (5.2, 0.75))]
    for a, b, rad, col, lab, lpos in edges:
        _arrow(axr, pos[a], pos[b], theme, color=col, lw=1.5, rad=rad,
               shrink=17)
        axr.text(lpos[0], lpos[1], lab, fontsize=6.4, color=col, ha="center",
                 va="center",
                 bbox=dict(boxstyle="round,pad=0.16", fc=theme.surface,
                           ec="none"))
    for name, p_ in pos.items():
        axr.scatter([p_[0]], [p_[1]], s=760, c=colour[name],
                    edgecolors=theme.surface, linewidths=1.8, zorder=4)
        axr.text(p_[0], p_[1], name, fontsize=8.5, color=theme.surface,
                 ha="center", va="center", zorder=5, fontweight="bold")
    axr.text(5.0, 5.6, "nodes = IP endpoints,  edges = individual flows",
             fontsize=7.2, color=theme.ink_secondary, fontweight="bold",
             ha="center")
    axr.text(5.0, -0.55,
             "parallel edges are kept, so every flow keeps its own attribute "
             "vector\nand receives its own prediction",
             fontsize=5.6, color=theme.ink_muted, ha="center")
    fig.suptitle("Flow records to a directed communication multigraph",
                 x=0.02, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, RESULTS_FIGURES / "fig02_flow_to_graph.png")


# --------------------------------------------------------------------------- #
# Figure 3 -- E-GAT-IDS architecture
# --------------------------------------------------------------------------- #


def figure3_architecture(theme: Theme = LIGHT) -> Path:
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.set_axis_off()

    stages = [
        ((1.1, 3.9), 1.8, 0.7, "Node features X\nendpoint behaviour\n(train flows only)",
         theme.categorical[0]),
        ((1.1, 1.3), 1.8, 0.7, "Edge features Z\nflow attributes", theme.categorical[2]),
        ((3.5, 3.9), 1.7, 0.6, "Node projection\nW·X", theme.ink_muted),
        ((6.0, 3.9), 2.0, 0.8, "Edge-aware GAT\nL layers, H heads\nα(hᵢ, hⱼ, zᵢⱼ)",
         theme.categorical[1]),
        ((8.6, 2.6), 1.8, 0.7, "Edge fusion\nrᵢⱼ = [hᵢ ‖ hⱼ ‖ zᵢⱼ]", theme.ink_muted),
        ((10.9, 2.6), 1.5, 0.7, "MLP\nsoftmax\nŷᵢⱼ", theme.categorical[5]),
    ]
    centres = []
    for (xy, w, h, label, colour) in stages:
        _box(ax, xy, w, h, label, theme, face=colour, ink=theme.surface,
             fontsize=5.8, radius=0.09)
        centres.append(xy)

    _arrow(ax, centres[0], centres[2], theme, lw=1.1)
    _arrow(ax, centres[2], centres[3], theme, lw=1.1)
    _arrow(ax, centres[3], centres[4], theme, lw=1.1)
    _arrow(ax, centres[4], centres[5], theme, lw=1.1)
    # edge features feed the attention, the message and the classifier
    _arrow(ax, centres[1], centres[3], theme, color=theme.categorical[2],
           lw=1.2, rad=0.18)
    _arrow(ax, centres[1], centres[4], theme, color=theme.categorical[2],
           lw=1.2, rad=-0.1)
    ax.annotate("into the attention logit\nand the message", (4.35, 2.35),
                fontsize=5.6, color=theme.categorical[2], ha="center")

    ax.annotate("message passing over training edges only", (6.0, 4.75),
                fontsize=6, color=theme.ink_muted, ha="center")
    ax.set_title("E-GAT-IDS: edge-level classification on the communication graph",
                 loc="left", fontsize=8.5, pad=4)
    return _save(fig, RESULTS_FIGURES / "fig03_architecture.png")


# --------------------------------------------------------------------------- #
# Figure 4 -- standard vs edge-aware attention
# --------------------------------------------------------------------------- #


def figure4_attention_contrast(theme: Theme = LIGHT) -> Path:
    apply_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45))
    titles = ["Standard GAT", "Edge-aware GAT (proposed placement)"]
    formulas = [
        r"$e_{ij}=\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}"
        r"[\mathbf{W}h_i \,\Vert\, \mathbf{W}h_j]\right)$",
        r"$e_{ij}=\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}"
        r"[\mathbf{W}h_i \,\Vert\, \mathbf{W}h_j \,\Vert\,"
        r" \mathbf{U}z_{ij}]\right)$",
    ]
    for k, (ax, title, formula) in enumerate(zip(axes, titles, formulas)):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
        centre = (5.0, 3.1)
        ax.scatter([centre[0]], [centre[1]], s=620, c=theme.categorical[0],
                   edgecolors=theme.surface, linewidths=1.6, zorder=4)
        ax.text(centre[0], centre[1], "$h_j$", fontsize=8,
                color=theme.surface, ha="center", va="center", zorder=5)
        neigh = [(1.6, 5.0), (1.2, 3.1), (1.6, 1.2)]
        widths = [2.4, 1.1, 0.5] if k == 1 else [1.5, 1.5, 1.5]
        for n, (p, w) in enumerate(zip(neigh, widths)):
            ax.scatter([p[0]], [p[1]], s=380, c=theme.categorical[2],
                       edgecolors=theme.surface, linewidths=1.4, zorder=4)
            ax.text(p[0], p[1], f"$h_{{i_{n+1}}}$", fontsize=6.5,
                    color=theme.surface, ha="center", va="center", zorder=5)
            _arrow(ax, p, centre, theme,
                   color=theme.categorical[1] if k == 1 else theme.ink_muted,
                   lw=w, shrink=13)
            mid = ((p[0] + centre[0]) / 2, (p[1] + centre[1]) / 2)
            if k == 1:
                ax.text(mid[0], mid[1] + 0.34, f"$z_{{i_{n+1}j}}$", fontsize=5.8,
                        color=theme.categorical[2], ha="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc=theme.surface,
                                  ec="none"))
        ax.text(5.0, 5.55, title, fontsize=7.5, color=theme.ink_primary,
                ha="center", fontweight="bold")
        ax.text(5.0, 0.55, formula, fontsize=6.4, color=theme.ink_secondary,
                ha="center")
        note = ("attention weight depends on the two endpoint states only"
                if k == 0 else
                "the flow's own attributes modulate how much it is attended to")
        ax.text(5.0, 0.02, note, fontsize=5.8, color=theme.ink_muted,
                ha="center")
        if k == 1:
            ax.annotate("arrow width ∝ α", (8.6, 4.6), fontsize=5.6,
                        color=theme.ink_muted, ha="center")
    fig.suptitle("Where the flow attributes enter the attention computation",
                 x=0.02, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, RESULTS_FIGURES / "fig04_attention_contrast.png")


def main() -> None:
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    roles = infer_roles()
    log.info("endpoint roles: %s", roles["counts"])
    figure1_testbed(roles)
    figure2_flow_to_graph()
    figure3_architecture()
    figure4_attention_contrast()


if __name__ == "__main__":
    main()
