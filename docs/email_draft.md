# 교수님께 보낼 주제 이메일 초안

> **마감: 2026-11-15(일) 오후 11:59** · 받는 사람: mzislam@sejong.ac.kr
> 공지에서 요구한 4항목(Team Name, Team Members, Project Title, Problem Statement)을 공지 순서대로 넣음.
> `[ ]` 부분 3곳(이름 2곳, 학번 1곳)을 채워서 **영어본**을 보낼 것. 한국어본은 내용 확인용.

---

## English (보낼 본문)

**To:** mzislam@sejong.ac.kr
**Subject:** [IoT Project] Project Information – StepCue (Individual) – [Full Name] [Student ID]

Dear Professor,

I am submitting my project information. I will work on the project individually.

1. **Team Name:** StepCue
2. **Team Members:** [Full Name] ([Student ID]), individual
3. **Project Title:** StepCue – An Edge AI Agent for Freezing-of-Gait Cueing in Parkinson's Disease
4. **Problem Statement:** People with Parkinson's disease often experience freezing of gait (FoG): while walking, their feet suddenly feel stuck to the floor, which can lead to falls. A rhythmic cue (a metronome sound or vibration) helps them start walking again, but it must be given within about one second and should match the person's own walking pace. A cloud-only system may be too slow or unreliable for this, and a cue given when the person stops on purpose is only a disturbance. I want to build an edge AI agent that detects FoG in real time and decides by itself when to start, strengthen, or stop a cue, when to notify a caregiver, and when not to act.

**Additional details**
- **Dataset:** Daphnet Freezing of Gait (from the whar-datasets list in the lecture slides): 3 accelerometers, 64 Hz, 10 subjects.
- **System:** A simulator replays the dataset over MQTT → a lightweight edge model detects FoG → an agent chooses an action → a Streamlit dashboard shows the results.
- **Hardware:** I do not have an edge device, so I will simulate the edge node on my laptop and measure model size and inference latency.
- **Repository:** https://github.com/bean0422ii-arch/stepcue

Thank you.

Best regards,
[Full Name]

---

## 한국어 (확인용, 보내지 않음)

**받는 사람:** mzislam@sejong.ac.kr
**제목:** [IoT 프로젝트] 프로젝트 정보 제출: StepCue (개인) – [이름] [학번]

교수님께,

프로젝트 정보를 제출합니다. 개인으로 진행하겠습니다.

1. **팀 이름:** StepCue
2. **팀원:** [이름] ([학번]), 개인
3. **프로젝트 제목:** StepCue: 파킨슨 보행동결 큐잉을 위한 엣지 AI 에이전트
4. **문제 정의:** 파킨슨 환자는 걷다가 발이 바닥에 붙은 듯 멈추는 보행동결(FoG)을 자주 겪고, 이때 넘어질 위험이 큽니다. 리듬 신호(메트로놈 소리나 진동)가 다시 걷는 데 도움이 되지만, 약 1초 안에 그 사람의 걸음 속도에 맞춰 줘야 합니다. 클라우드만 쓰면 너무 느리거나 불안정할 수 있고, 일부러 멈춰 섰을 때 울리는 신호는 방해만 됩니다. 그래서 엣지에서 동결을 실시간으로 감지하고, 큐 시작·강화·중지, 보호자 알림, 그리고 행동하지 않을 때를 스스로 판단하는 AI 에이전트를 만들려고 합니다.

**추가 정보**
- **데이터셋:** Daphnet Freezing of Gait (수업 자료의 whar-datasets 목록): 가속도계 3개, 64Hz, 환자 10명
- **시스템:** 시뮬레이터가 데이터를 MQTT로 재생 → 경량 엣지 모델이 동결 감지 → 에이전트가 행동 결정 → Streamlit 대시보드에 표시
- **하드웨어:** 엣지 디바이스가 없어서 노트북에서 엣지 노드를 시뮬레이션하고, 모델 크기와 추론 지연을 측정합니다.
- **저장소:** https://github.com/bean0422ii-arch/stepcue

감사합니다.

[이름] 드림
