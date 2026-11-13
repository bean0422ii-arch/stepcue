# 교수님께 보낼 주제 이메일 초안

> 다음 월요일 수업(11/16) 전까지 발송. 슬라이드가 영어라 영어본을 먼저 넣었고, 한국어본도 함께 둠.
> `[ ]` 부분을 채워서 보낼 것.

---

## English

**Subject:** [IoT Project] Topic Submission – StepCue (Individual) – [Name] [Student ID]

Dear Professor [Name],

I would like to submit my AIoT project topic. I will work on the project individually.

- **Name / Student ID:** [Name] / [Student ID]
- **Project title:** StepCue – An Edge AI Agent for Freezing-of-Gait Cueing in Parkinson's Disease
- **Problem:** People with Parkinson's disease often experience freezing of gait (FoG) while walking, which can lead to falls. Rhythmic cues (a metronome sound or vibration) help them start walking again, but the cue must be given within about one second. A cloud-only solution may be too slow or unreliable, so detection and response should happen on the edge.
- **Dataset:** Daphnet Freezing of Gait (from the whar-datasets list in the lecture slides): 3 accelerometers, 64 Hz, 10 subjects.
- **System:** A sensor simulator replays the dataset over MQTT → a lightweight edge model detects FoG in real time → an agent decides whether to start, escalate, or stop the cue, or to notify a caregiver → a Streamlit dashboard shows the results.
- **Agent behavior:** The agent does not act when a person stops walking on purpose. It measures each person's normal walking cadence and sets the cue tempo to match it, and it changes the tempo and adds vibration if the person does not recover within 3 seconds.
- **Hardware:** I do not have an edge device, so I will simulate the edge node on my laptop and measure model size and inference latency to show that it is suitable for the edge.

Thank you. I will present my progress at the next progress meeting.

Best regards,
[Name]

---

## 한국어

**제목:** [IoT 프로젝트] 주제 제출: StepCue (개인) – [이름] [학번]

[교수님 성함] 교수님께,

AIoT 프로젝트 주제를 제출합니다. 개인으로 진행하겠습니다.

- **이름 / 학번:** [이름] / [학번]
- **프로젝트 제목:** StepCue: 파킨슨 보행동결 큐잉을 위한 엣지 AI 에이전트
- **문제:** 파킨슨 환자는 걷다가 보행동결(FoG)을 자주 겪고, 이때 넘어질 위험이 큽니다. 리듬 신호(메트로놈 소리나 진동)가 다시 걷는 데 도움이 되지만 1초 안에 주어야 합니다. 클라우드만 쓰면 너무 느리거나 불안정할 수 있어서 엣지에서 감지하고 대응하려고 합니다.
- **데이터셋:** Daphnet Freezing of Gait (수업 자료의 whar-datasets 목록): 가속도계 3개, 64Hz, 환자 10명
- **시스템:** 센서 시뮬레이터가 데이터를 MQTT로 재생 → 경량 엣지 모델이 실시간으로 동결 감지 → 에이전트가 큐 시작·강화·중지 또는 보호자 알림을 결정 → Streamlit 대시보드에 표시
- **에이전트 동작:** 사람이 일부러 멈춰 선 경우에는 행동하지 않습니다. 또 그 사람의 평소 걸음 속도를 측정해 큐 템포를 맞추고, 3초 안에 회복하지 않으면 템포를 바꾸고 진동을 추가합니다.
- **하드웨어:** 엣지 디바이스가 없어서 노트북에서 엣지 노드를 시뮬레이션하고, 모델 크기와 추론 지연을 측정해 엣지 적합성을 보이겠습니다.

다음 진척 점검 때 진행 상황을 발표하겠습니다. 감사합니다.

[이름] 드림
