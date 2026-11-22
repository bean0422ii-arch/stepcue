"""13주차: 엣지 vs 클라우드 지연 비교 실험. "왜 엣지에서 판단해야 하나?"

두 경로 (판단 방법은 같고, 판단하는 장소만 다름)
  엣지:     센서 -> 착용 기기에서 판단 -> 큐                         지연 = 처리 시간(실측 약 3ms)
  클라우드: 센서 -> 인터넷 -> 서버에서 판단 -> 인터넷 -> 기기에서 큐   지연 = 처리 시간 + 네트워크 왕복 200~800ms

방법
- 237개 동결 에피소드마다, 12주차 LOSO 확률(3윈도우 평균 >= 0.8)로 "알람이 처음 켜지는 시각"을 구한다.
- 큐가 울리는 시각 = 알람 시각 + 경로 지연
- 지표
    반응 시간: 동결 시작 -> 큐 (중앙값)
    동결 중 도착: 큐가 동결이 끝나기 전에 도착한 비율 (끝난 뒤 울리는 큐는 도움이 안 됨)
    짧은 동결(3초 이하)에서의 동결 중 도착 비율
- 데이터 양: 클라우드는 원시 가속도(64Hz)를 모두 올려야 하고, 엣지는 0.5초마다 요약만 보낸다.

한계: 네트워크 지연은 실제로 재지 않고 고정값을 더했다(실제 망은 지연이 흔들리고 끊기기도 함 -> 클라우드에 더 불리).

실행: .venv\\Scripts\\python.exe src\\cloud_latency.py   (먼저 src\\calibrate.py로 LOSO 확률 저장 필요)
결과: docs/cloud_latency.md, docs/figures/cloud_latency.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate import loso_scores
from data import ACC_COLS, FS, ROOT, load_run
from train import DETECT_GRACE, THRESHOLD, label_episodes

EDGE_MS = 3                            # 엣지 실측: 센서 샘플 발행 -> 판단 완료 2~4ms (docs/edge_metrics.md)
PATHS = {"엣지": EDGE_MS, "클라우드 200ms": 200, "클라우드 400ms": 400,
         "클라우드 600ms": 600, "클라우드 800ms": 800}
SHORT = 3.0                            # 짧은 동결 기준(초)


def first_alarms(w: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """에피소드마다 첫 알람 시각(윈도우 끝). 못 잡았으면 NaN."""
    w = w.assign(alarm=w["avg"] >= THRESHOLD)
    out = []
    for ep in episodes.itertuples():
        g = w[(w.subject == ep.subject) & (w.run == ep.run) & w.alarm
              & (w.t_end >= ep.start) & (w.t_end <= ep.end + DETECT_GRACE)]
        out.append(g.t_end.iloc[0] if len(g) else np.nan)
    return episodes.assign(alarm=out, length=episodes.end - episodes.start)


def compare(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    detected = ep.dropna(subset=["alarm"])
    short = ep.length <= SHORT
    for name, ms in PATHS.items():
        cue = ep.alarm + ms / 1000                       # 큐가 실제로 울리는 시각
        in_time = (cue <= ep.end).fillna(False)          # 못 잡은 에피소드는 "도착 못 함"
        reaction = (detected.alarm + ms / 1000 - detected.start)
        rows.append({"경로": name, "지연(ms)": ms,
                     "반응 시간 중앙값(초)": reaction.median(),
                     "동결 중 도착(전체)": in_time.mean(),
                     f"동결 중 도착({SHORT:.0f}초 이하 동결)": in_time[short].mean(),
                     "1초 안에 큐": (reaction <= 1.0).sum() / len(ep)})
    return pd.DataFrame(rows).set_index("경로")


def bandwidth() -> pd.DataFrame:
    """실제 메시지 형식(JSON)으로 초당 전송량 계산."""
    df = load_run("S01", "R01").iloc[:FS * 60]
    imu = [len(json.dumps({"t": round(i / FS, 4), "acc": r.tolist(), "label": 1, "sent": 1.7e9 + i / FS}))
           for i, r in enumerate(df[ACC_COLS].to_numpy(dtype=float))]
    event = len(json.dumps({"t": 1144.48, "fog_prob": 0.912, "prob_avg": 0.853, "fog": True, "threshold": 0.8,
                            "walking": True, "rhythm": 0.62, "cadence": 151, "infer_ms": 0.62,
                            "latency_ms": 2.2, "truth": 1}))
    cloud, edge = np.mean(imu) * FS, event * 2           # 바이트/초
    return pd.DataFrame({"보내는 것": ["원시 가속도 9채널 × 64Hz", "요약 이벤트 × 2Hz"],
                         "초당 메시지": [FS, 2], "초당 데이터(KB)": [cloud / 1024, edge / 1024],
                         "하루 8시간(MB)": [cloud * 3600 * 8 / 1024 ** 2, edge * 3600 * 8 / 1024 ** 2]},
                        index=["클라우드 (올려야 하는 것)", "엣지 (밖으로 나가는 것)"])


def plot(table: pd.DataFrame):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#2a9d8f"] + ["#e76f51"] * (len(table) - 1)
    labels = [n.replace("클라우드 ", "클라우드\n") for n in table.index]
    a1.bar(labels, table["반응 시간 중앙값(초)"], color=colors)
    a1.set_title("동결 시작 → 큐 (반응 시간 중앙값)")
    a1.set_ylabel("초")
    for i, v in enumerate(table["반응 시간 중앙값(초)"]):
        a1.text(i, v + 0.03, f"{v:.2f}", ha="center")
    x = table["지연(ms)"]
    a2.plot(x, table["동결 중 도착(전체)"] * 100, "o-", label="전체 동결")
    a2.plot(x, table[f"동결 중 도착({SHORT:.0f}초 이하 동결)"] * 100, "s-", label=f"{SHORT:.0f}초 이하 짧은 동결")
    a2.set_title("큐가 동결이 끝나기 전에 도착한 비율")
    a2.set_xlabel("판단 경로 지연(ms)  (맨 왼쪽 = 엣지)")
    a2.set_ylabel("%")
    a2.set_ylim(0, 100)
    a2.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "docs" / "figures" / "cloud_latency.png", dpi=120)
    plt.close()


def main():
    ep = first_alarms(loso_scores(), label_episodes())
    table = compare(ep)
    bw = bandwidth()
    print(table.round(3).to_string())
    print(bw.round(2).to_string())
    plot(table)

    t = table.copy()
    for c in t.columns:
        if "도착" in c or "1초" in c:
            t[c] = (t[c] * 100).round(1).astype(str) + "%"
    e, c8 = table.iloc[0], table.loc["클라우드 800ms"]
    short_col = f"동결 중 도착({SHORT:.0f}초 이하 동결)"
    md = ["# 엣지 vs 클라우드 지연 비교 (13주차)", "",
          "`src/cloud_latency.py`로 생성. 질문: **판단을 클라우드에서 하면 무엇이 나빠지는가?**", "",
          "- 두 경로의 판단 방법은 같다(같은 모델, 임계값 0.8, 3윈도우 평균). 판단하는 장소만 다르다.",
          f"- 엣지 지연 {EDGE_MS}ms는 실측값([edge_metrics.md](edge_metrics.md)). 클라우드는 네트워크 왕복 지연 200~800ms를 더함",
          f"- 동결 에피소드 {len(ep)}개 (12주차 LOSO 확률로 감지한 {ep.alarm.notna().sum()}개 기준 반응 시간). "
          f"{SHORT:.0f}초 이하 짧은 동결 {int((ep.length <= SHORT).sum())}개", "",
          "## 1. 반응 시간과 동결 중 도착", "", t.round(2).to_markdown(), "",
          "![엣지 vs 클라우드](figures/cloud_latency.png)", "",
          "- 반응 시간 = 동결 시작 → 큐. 감지 자체에 걸리는 시간(약 1초, 2초 윈도우를 채워야 함)이 대부분이고, 경로 지연이 그 위에 더해진다",
          "- 동결 중 도착 = 큐가 동결이 끝나기 전에 울린 비율. 끝난 뒤 울리는 큐는 도움이 안 되고 오히려 혼란을 줌",
          "", "## 2. 보내야 하는 데이터 양 (실제 MQTT 메시지 형식 기준)", "", bw.round(2).to_markdown(), "",
          "## 3. 해석", "",
          f"- 클라우드 800ms면 반응 시간이 {e['반응 시간 중앙값(초)']:.2f}초 → {c8['반응 시간 중앙값(초)']:.2f}초로 늘고, "
          f"동결 중에 큐가 도착하는 비율이 {e['동결 중 도착(전체)']:.0%} → {c8['동결 중 도착(전체)']:.0%}로 줄어든다.",
          f"- 특히 **{SHORT:.0f}초 이하 짧은 동결**에서 차이가 크다: {e[short_col]:.0%} → {c8[short_col]:.0%}. "
          "동결 길이의 중앙값이 5.3초로 짧아서, 몇백 ms 차이가 큐가 쓸모 있는지를 가른다.",
          f"- 데이터 양: 엣지는 밖으로 보내는 양이 약 {bw.iloc[0, 2] / bw.iloc[1, 2]:.0f}분의 1이다. "
          "원시 움직임 데이터가 밖으로 나가지 않아 개인정보 면에서도 유리하다.",
          "- 네트워크가 끊기면 클라우드 경로는 아예 큐를 줄 수 없다. 엣지는 혼자 계속 동작한다(14주차 안전 대비 동작).",
          "- 그래서 **실시간 판단(빠른 층)은 엣지**, 세션이 끝난 뒤의 분석과 임계값 조정(느린 층, LLM)은 **클라우드**에 둔다. "
          "느린 층은 몇 초가 걸려도 큐 타이밍에 영향이 없다.", "",
          "## 4. 한계", "",
          "- 네트워크 지연은 실제로 재지 않고 고정값을 더했다. 실제 망은 지연이 흔들리고(지터) 끊기기도 해서 클라우드에 더 불리하다.",
          "- 서버의 처리 시간은 엣지와 같다고 가정했다(서버가 더 빨라도 0.07ms 수준이라 차이 없음).", ""]
    (ROOT / "docs" / "cloud_latency.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
