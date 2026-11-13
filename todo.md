# StepCue: 5주 TODO (11~15주차)

> 기간: 2026-11-11(수, 11주차 1차 수업) ~ **2026-12-14(15주차 2차 수업)**, 총 5주(교수님 공지 기준).
> 주차 날짜는 월~일 기준으로 추정한 값이니 실제 수업 일정에 맞춰 조정할 것.
> 매주 **2차 수업 = 진척 점검(Progress Meeting)**. 아래 "점검 산출물"을 그 전까지 준비.

---

## 11주차 (11/9 ~ 11/15): 주제 확정 · 데이터 이해
- [x] 교수님께 주제 이메일 보내기, **11/15(일) 오후 11:59까지** (mzislam@sejong.ac.kr) → [docs/email_draft.md](docs/email_draft.md)
- [x] Daphnet FoG 데이터셋 다운로드 → `data/raw/` (UCI 저장소)
- [x] 데이터 탐색: 환자별 레이블 분포(0 실험 외 / 1 정상 / 2 동결), 동결 구간 그래프 → [docs/data_summary.md](docs/data_summary.md)
- [x] 동결이 없는 환자(S04, S10) 확인 → 오탐 테스트용으로 따로 표시
- [x] Problem card 마무리 → [docs/problem_card.md](docs/problem_card.md)
- [x] 1페이지 설계 작성(데이터, 이벤트, 에이전트 도구, 행동) → [docs/design.md](docs/design.md)
- [x] Git 저장소 초기화, `.gitignore` 작성
- [x] GitHub 원격 저장소 연결 후 push → https://github.com/bean0422ii-arch/stepcue
- [x] 시스템 다이어그램 생성 → [docs/figures/system_diagram.png](docs/figures/system_diagram.png) (슬라이드 12 프롬프트 채운 버전: [docs/diagram_prompt.md](docs/diagram_prompt.md))
- [ ] AI 사용 기록 시작 → `docs/ai_use.md` (사용한 도구, 프롬프트, AI가 틀린 점)

**점검 산출물:** Problem card, 1페이지 설계, 다이어그램, 데이터 탐색 그래프

## 12주차 (11/16 ~ 11/22): 모델
- [ ] `src/features.py`: 2초 윈도우(0.5초 간격) 분할, Freeze Index, 통계 특징 약 10개
- [ ] 베이스라인: Freeze Index에 임계값 적용 → 민감도, 특이도, F1
- [ ] `src/train.py`: RandomForest 또는 작은 MLP 학습, 환자 단위 교차검증(LOSO)
- [ ] 베이스라인과 모델 성능 비교표 작성
- [ ] 걸음 속도(cadence) 추정: 발목 가속도 피크 간격 → 환자별 분당 걸음 수 표
- [ ] `scripts/download_data.py`: Daphnet 자동 다운로드 (README 명령어 줄이기용)
- [ ] `src/simulator.py`: 녹화 데이터를 64Hz 실시간처럼 재생(아직 MQTT 없이 콘솔 출력)

**점검 산출물:** 성능표, 동작하는 시뮬레이터

## 13주차 (11/23 ~ 11/29): 엣지 파이프라인
- [ ] MQTT 브로커 실행 (Mosquitto 별도 설치 대신 파이썬 브로커로 대체 가능한지 검토, README 명령어 수 때문)
- [ ] 시뮬레이터 → `stepcue/imu` 발행
- [ ] `src/edge_node.py`: 구독 → 윈도우 → 추론 → `stepcue/event` 발행
- [ ] 모델을 ONNX로 변환, 모델 크기(KB)와 윈도우당 추론 시간(ms) 측정
- [ ] 클라우드 지연 비교 실험: 인위적 지연 200~800ms 넣은 경로와 반응 시간 비교 그래프

**점검 산출물:** 실시간 엣지 추론 데모, 엣지 지표표와 그래프

## 14주차 (11/30 ~ 12/6): 에이전트 · 대시보드 · end-to-end
- [ ] `src/agent.py`: 상태 머신(아무것도 안 함 / 큐 시작 / 큐 강화 / 보호자 알림 / 큐 중지)
- [ ] 큐 템포: ① 기본 템포 = 개인 걸음 속도, ② 3초 미회복 시 90%로 낮추고 진동 추가 (밴딧은 제외, design.md 5절)
- [ ] 모든 판단을 `logs/decisions.jsonl`에 기록(입력값, 행동, 이유)
- [ ] `dashboard/app.py` (Streamlit): 실시간 FoG 확률, 메트로놈 표시, 보호자 알림, 결정 로그
- [ ] `scenarios/`에 테스트 시나리오 3개 만들기
  - [ ] ① 동결 → 큐 → 회복 → 중지 (감지에서 큐까지 시간 ms 기록)
  - [ ] ② 긴 동결 → 큐 강화 → 보호자 알림
  - [ ] ③ 동결 없는 환자(S04)가 걷다 서기 반복 → **행동하지 않음**
- [ ] (선택) 합성 통계만 넣은 LLM 하루 요약

**점검 산출물:** end-to-end 동작, 시나리오 3개의 결정 로그

## 15주차 (12/7 ~ 12/13): 마무리 · 제출
- [ ] 버그 수정, 처음부터 다시 설치해서 실행되는지 테스트
- [ ] `README.md`: 명령어 **4개 이하**로 실행할 수 있게 정리 (수업 요구 "under 5 commands" = 5개 미만)
- [ ] `run_demo.py`: 브로커·시뮬레이터·엣지·에이전트·대시보드 한 번에 실행
- [ ] `docs/ai_use.md` 완성: 도구, 핵심 프롬프트 5개, AI가 틀린 것 1개와 발견 방법
- [ ] 최종 보고서(의료 진단용이 아니라 데이터셋 기반 시연이라는 점 명시)
- [ ] 슬라이드 5장: 사용자, 문제, 데모, 결과, 다음 단계
- [ ] 3분 데모 영상 녹화
- [ ] 코드 설명 연습: `features.py`의 Freeze Index, `agent.py`의 판단 로직, `edge_node.py`의 윈도우 처리
- [ ] GitHub 저장소 정리 후 제출

---

## 교수님께 확인할 것
- [x] 기간: 공지로 5주, 12/14(15주차 2차 수업) 종료 확인
- [ ] 세부 배점(슬라이드 17에 내용 없음)
- [ ] 최종 보고서 형식과 분량
