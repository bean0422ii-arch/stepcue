"""11주차 데이터 탐색: 환자별 레이블 분포, 동결 에피소드 통계, 동결 구간 그래프.

실행: .venv\\Scripts\\python.exe src\\explore.py
결과: docs/figures/*.png, docs/data_summary.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "dataset_fog_release" / "dataset"
FIG_DIR = ROOT / "docs" / "figures"
FS = 64  # Hz

COLUMNS = ["time_ms",
           "ankle_x", "ankle_y", "ankle_z",    # 발목(정강이)
           "thigh_x", "thigh_y", "thigh_z",    # 허벅지
           "trunk_x", "trunk_y", "trunk_z",    # 몸통(허리)
           "label"]                            # 0 실험 외 / 1 정상 / 2 동결


def load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(DATA_DIR.glob("S*R*.txt")):
        df = pd.read_csv(f, sep=" ", header=None, names=COLUMNS)
        df["subject"] = f.stem[:3]
        df["run"] = f.stem[3:]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def freeze_episodes(df_run: pd.DataFrame) -> list[float]:
    """한 run 안에서 레이블 2가 연속된 구간들의 길이(초)."""
    is_fog = (df_run["label"].to_numpy() == 2).astype(int)
    edges = np.diff(np.concatenate([[0], is_fog, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return list((ends - starts) / FS)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()

    # 1) 환자별 레이블 분포(분 단위)
    dist = (df.groupby(["subject", "label"]).size().unstack(fill_value=0) / FS / 60).round(1)
    dist = dist.rename(columns={0: "실험외(분)", 1: "정상(분)", 2: "동결(분)"})

    # 2) 동결 에피소드 통계
    rows = []
    all_eps = []
    for subj, g in df.groupby("subject"):
        eps = []
        for _, r in g.groupby("run"):
            eps += freeze_episodes(r)
        all_eps += eps
        rows.append({"subject": subj,
                     "에피소드 수": len(eps),
                     "평균 길이(초)": round(np.mean(eps), 1) if eps else 0,
                     "최장(초)": round(max(eps), 1) if eps else 0,
                     "10초 초과": sum(e > 10 for e in eps)})
    ep = pd.DataFrame(rows).set_index("subject")
    summary = dist.join(ep)
    summary["동결 비율(%)"] = (summary["동결(분)"] /
                              (summary["정상(분)"] + summary["동결(분)"]) * 100).round(1)
    print(summary.to_string())

    # 그림 1: 환자별 정상/동결 시간
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    ax = summary[["정상(분)", "동결(분)"]].plot.bar(stacked=True, figsize=(8, 4),
                                                   color=["#9db4c0", "#d1495b"])
    ax.set_title("환자별 정상 / 동결 시간 (분)")
    ax.set_ylabel("분")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_distribution.png", dpi=120)
    plt.close()

    # 그림 2: 동결 에피소드 길이 분포
    plt.figure(figsize=(7, 4))
    plt.hist(all_eps, bins=40, color="#d1495b")
    plt.axvline(3, ls="--", c="k", lw=1, label="3초 (큐 강화 기준)")
    plt.axvline(10, ls=":", c="k", lw=1, label="10초 (보호자 알림 기준)")
    plt.title(f"동결 에피소드 길이 분포 (총 {len(all_eps)}개)")
    plt.xlabel("초")
    plt.ylabel("개수")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "episode_lengths.png", dpi=120)
    plt.close()

    # 그림 3: 동결 구간 예시 (S01R01 첫 동결 앞뒤 20초, 발목 가속도)
    r = df[(df.subject == "S01") & (df.run == "R01")].reset_index(drop=True)
    first = int(np.argmax(r["label"].to_numpy() == 2))
    seg = r.iloc[max(0, first - 10 * FS): first + 20 * FS]
    t = (seg["time_ms"] - seg["time_ms"].iloc[0]) / 1000
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for c in ["ankle_x", "ankle_y", "ankle_z"]:
        axes[0].plot(t, seg[c], lw=0.7, label=c)
    fog = seg["label"].to_numpy() == 2
    axes[0].fill_between(t, seg[["ankle_x", "ankle_y", "ankle_z"]].min().min(),
                         seg[["ankle_x", "ankle_y", "ankle_z"]].max().max(),
                         where=fog, color="#d1495b", alpha=0.2, label="동결(레이블 2)")
    axes[0].set_ylabel("가속도 (mg)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("S01R01 동결 구간 예시: 발목 가속도")
    axes[1].plot(t, seg["label"], c="k", drawstyle="steps-post")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_ylabel("레이블")
    axes[1].set_xlabel("초")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fog_example_S01.png", dpi=120)
    plt.close()

    # 요약 저장
    no_fog = summary.index[summary["에피소드 수"] == 0].tolist()
    md = ["# 데이터 탐색 요약 (11주차)", "",
          "`src/explore.py`로 생성. 샘플링 64Hz 기준으로 분 단위 환산.", "",
          summary.to_markdown(), "",
          f"- 전체 동결 에피소드: {len(all_eps)}개, 중앙값 {np.median(all_eps):.1f}초, "
          f"3초 초과 {sum(e > 3 for e in all_eps)}개, 10초 초과 {sum(e > 10 for e in all_eps)}개",
          f"- 동결이 없는 환자: {', '.join(no_fog)} → 시나리오 ③(행동하지 않음) 테스트용", "",
          "![레이블 분포](figures/label_distribution.png)",
          "![에피소드 길이](figures/episode_lengths.png)",
          "![동결 예시](figures/fog_example_S01.png)", ""]
    (ROOT / "docs" / "data_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
