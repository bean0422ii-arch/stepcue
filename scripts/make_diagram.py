"""StepCue 시스템 다이어그램 생성.

실행: .venv\\Scripts\\python.exe scripts\\make_diagram.py
결과: docs/figures/system_diagram.png, docs/figures/system_diagram.svg
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

# 파스텔 팔레트: (배경, 테두리)
WEAR = ("#E3F0FB", "#5B8DB8")
EDGE = ("#E6F4EA", "#4E9A6B")
AGENT = ("#FFF1E0", "#D08A3C")
OUTP = ("#F1E8F7", "#8E6BB0")
LLM = ("#FDF3E7", "#B5651D")
INK = "#2B2B2B"
SUB = "#5F5F5F"


def box(ax, x, y, w, h, colors, title=None, body=None, *, title_size=11,
        body_size=8.6, dashed=False, lw=1.4, radius=0.18, z=2, fill=None):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fill or face, ec=edge, lw=lw,
                                ls=(0, (4, 3)) if dashed else "-", zorder=z))
    if title and body:
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                fontsize=title_size, weight="bold", color=INK, zorder=z + 1)
        ax.text(x + w / 2, y + h - 0.66, body, ha="center", va="top",
                fontsize=body_size, color=SUB, linespacing=1.35, zorder=z + 1)
    elif title:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=INK, zorder=z + 1)


def group(ax, x, y, w, h, colors, label):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                fc=face, ec=edge, lw=1.8, zorder=1, alpha=0.55))
    ax.text(x + 0.2, y + h - 0.18, label, ha="left", va="top",
            fontsize=12.5, weight="bold", color=edge, zorder=2)


def arrow(ax, p, q, color=INK, label=None, label_xy=None, dashed=False,
          lw=1.6, rad=0.0, label_color=None, fs=8.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw,
                                 ls=(0, (4, 3)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))
    if label:
        lx, ly = label_xy or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.22)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=fs,
                family="monospace", color=label_color or color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(8, 8.7, "StepCue: Edge AI Agent for Freezing-of-Gait Cueing",
            ha="center", va="center", fontsize=16, weight="bold", color=INK)

    # ── 1. Wearable (simulated) ─────────────────────────────
    group(ax, 0.25, 2.0, 2.7, 6.1, WEAR, "Wearable (sim)")
    box(ax, 0.45, 4.9, 2.3, 2.0, WEAR, "Daphnet replay",
        "3 accelerometers\nankle · thigh · trunk\n64 Hz, 10 subjects", fill="white")
    box(ax, 0.45, 2.3, 2.3, 2.2, WEAR, "Real-time simulator",
        "recorded data\nreplayed at 64 Hz\nas if live", title_size=10,
        fill="white")
    arrow(ax, (1.6, 4.9), (1.6, 4.5))

    # ── 2. Edge node ─────────────────────────────────────────
    group(ax, 3.75, 2.0, 4.5, 6.1, EDGE, "Edge node (laptop)")
    box(ax, 4.05, 5.75, 3.9, 1.15, EDGE, "Windowing",
        "2 s window · 0.5 s hop", fill="white")
    box(ax, 4.05, 4.25, 3.9, 1.2, EDGE, "Features (10)",
        "Freeze Index · band power · dominant freq\n+ walking rhythm & cadence (last 10 s)", fill="white")
    box(ax, 4.05, 2.75, 3.9, 1.2, EDGE, "FoG model (ONNX, 70 KB)",
        "RandomForest, 20 trees\nfog_prob ≥ threshold (config/user.json)", fill="white")
    arrow(ax, (6.0, 5.75), (6.0, 5.45))
    arrow(ax, (6.0, 4.25), (6.0, 3.95))
    ax.text(6.0, 2.35, "measured: 0.07 ms / window · sensor → decision 2–4 ms",
            ha="center", va="center", fontsize=8.2, color=EDGE[1], style="italic")

    # ── 3. Agent (fast layer) ────────────────────────────────
    group(ax, 9.05, 2.0, 4.0, 6.1, AGENT, "Agent: fast layer (< 1 s)")
    box(ax, 9.3, 5.75, 3.5, 1.15, AGENT, "Context check",
        "walking in last 5 s?  episode length?", fill="white")
    box(ax, 9.3, 4.25, 3.5, 1.2, AGENT, "State machine",
        "IDLE → CUEING → ESCALATED\nthresholds: 3 s · 10 s · 3 / 10 min", fill="white")
    box(ax, 9.3, 2.75, 3.5, 1.2, AGENT, "Tools",
        "play_cue(bpm) · stop_cue()\nnotify_caregiver() · log_decision()", fill="white")
    arrow(ax, (11.05, 5.75), (11.05, 5.45))
    arrow(ax, (11.05, 4.25), (11.05, 3.95))
    ax.text(11.05, 2.35, "cue tempo = personal cadence  (×0.9 on escalate)",
            ha="center", va="center", fontsize=8.2, color=AGENT[1], style="italic")

    # ── 4. Outputs ───────────────────────────────────────────
    group(ax, 13.85, 2.0, 1.95, 6.1, OUTP, "Outputs")
    outs = [(5.7, "Patient", "metronome\n+ vibration"),
            (4.25, "Caregiver", "alert"),
            (2.8, "Dashboard", "Streamlit\n+ decisions.jsonl")]
    for y, t, b in outs:
        box(ax, 14.0, y, 1.65, 1.2, OUTP, t, b, title_size=10, body_size=8,
            fill="white")

    # ── 메인 흐름 화살표(MQTT 토픽) ──────────────────────────
    arrow(ax, (2.75, 3.4), (4.05, 6.3), label="stepcue/imu",
          label_xy=(3.35, 5.05), rad=-0.15)
    arrow(ax, (7.95, 3.35), (9.3, 6.3), label="stepcue/event",
          label_xy=(8.6, 5.05), rad=-0.15)
    for y in (6.3, 4.85, 3.4):
        arrow(ax, (12.8, 3.35), (14.0, y), rad=0.0, lw=1.3)
    ax.text(13.42, 6.0, "stepcue/\naction", ha="center", va="center",
            fontsize=8.4, family="monospace", color=INK,
            bbox=dict(fc="white", ec="none", pad=1.2), zorder=6)

    # ── 핵심 포인트(novelty) 배지 ────────────────────────────
    box(ax, 9.3, 7.0, 3.5, 0.5, ("#FFFFFF", "#C0392B"),
        "Knows when NOT to act", title_size=9.5, lw=1.6, radius=0.12, z=6)
    box(ax, 4.05, 7.0, 3.9, 0.5, ("#FFFFFF", "#4E9A6B"),
        "Raw IMU never leaves the edge", title_size=9.5, lw=1.6, radius=0.12, z=6)

    # ── 느린 층: LLM 보정 에이전트 (실시간 경로 밖) ─────────────
    box(ax, 5.2, 0.15, 7.4, 1.45, LLM,
        "Agent: slow layer — LLM calibration (Claude API, after each session)",
        "reads log summaries + \"Wrong cue\" feedback (never raw IMU) → finds patterns\n"
        "simulate_threshold() · set_threshold() · request_approval() · update_memory()",
        dashed=True, title_size=10, body_size=8.4, fill="white")
    arrow(ax, (14.8, 2.8), (12.6, 0.95), color=LLM[1], dashed=True, rad=-0.25,
          label="decisions.jsonl\nfeedback.jsonl", label_xy=(14.35, 1.0), fs=8)
    arrow(ax, (5.2, 0.95), (4.6, 2.75), color=LLM[1], dashed=True, rad=-0.3,
          label="new threshold\n(human approval\nif large)", label_xy=(3.55, 1.25), fs=8)

    # 범례·클라우드 비교 결과
    ax.text(0.3, 0.45, "All links: MQTT (local broker)\n"
            "Cloud path test (+200–800 ms):\ncue 1.06 s → 1.85 s", ha="left",
            va="center", fontsize=8.2, color=SUB)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"system_diagram.{ext}", dpi=160, bbox_inches="tight",
                    facecolor="white")
    print("saved:", OUT / "system_diagram.png")


if __name__ == "__main__":
    main()
