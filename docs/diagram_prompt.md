# 시스템 다이어그램 프롬프트 (수업 슬라이드 12 양식)

> 기본 다이어그램은 코드로 생성: `scripts/make_diagram.py` → [figures/system_diagram.png](figures/system_diagram.png)
> 아래 프롬프트는 슬라이드 12 양식에 StepCue 내용을 채운 것. 이미지 생성 AI(Gemini, ChatGPT 등)에 그대로 붙여 넣어 발표용 일러스트를 만들 때 사용.
> AI가 만든 그림은 글자가 틀리는 경우가 많으니 라벨을 꼭 확인할 것. 사용했다면 `ai_use.md`에 기록.

---

You are an expert Scientific Illustrator for top-tier AI conferences (NeurIPS/CVPR/ICML).
Your task is to generate a professional "Illustration" (main figure for the paper) based on a research paper abstract and methodology.

**Abstract:**
People with Parkinson's disease often experience freezing of gait (FoG): their feet suddenly feel stuck to the floor while walking, which can cause falls. A rhythmic cue matched to the person's walking pace helps them start walking again, but it must be given within about one second. StepCue is an edge AI agent that detects FoG from wearable accelerometers in real time and decides by itself whether to start, strengthen, or stop a metronome cue, whether to notify a caregiver, and when not to act (for example, when the person stops on purpose). Raw sensor data never leaves the edge; only events are sent. We compare the edge path with a cloud path that adds 200–800 ms of delay.

**Methodology:**
1. Wearable (simulated): the Daphnet dataset (3 accelerometers on ankle, thigh, trunk; 64 Hz) is replayed in real time over MQTT (topic `stepcue/imu`).
2. Edge node: 2 s sliding windows (0.5 s hop) → features (Freeze Index = 3–8 Hz / 0.5–3 Hz power ratio, walking cadence, energy, variance) → small ONNX classifier (RandomForest or small MLP) → FoG probability, published as `stepcue/event`. Model size (KB) and latency (ms per window) are measured.
3. Agent: context check (was the person walking in the last 5 s?) → state machine IDLE → CUEING → ESCALATED → tools: play_cue(bpm), stop_cue(), notify_caregiver(), log_decision(). Cue tempo = the person's own walking cadence; on escalation the tempo is lowered to 90% and vibration is added. Actions are published as `stepcue/action`.
4. Outputs: patient (metronome + vibration), caregiver alert, Streamlit dashboard with a decision log.
5. Cloud baseline (dashed, gray): the same model with an artificial 200–800 ms delay, used only for comparing detection-to-cue time.

**Visual Style Requirements:**
1.  **Style:** Flat vector illustration, clean lines, academic aesthetic. Similar to figures in DeepMind or OpenAI papers.
2.  **Layout:** Organized flow (Left-to-Right, Top-to-Bottom, Circular and other shapes). Group related components logically.
3.  **Color Palette:** Professional pastel tones. White background.
4.  **Text Rendering:** You MUST include legible text labels for key modules or equations mentioned in the methodology (e.g., "Encoder", "Loss", "Transformer").
5.  **Negative Constraints:** NO photorealistic photos, NO messy sketches, NO unreadable text, NO 3D shading artifacts.

**Generation Instruction:**
Highlight the core novelty. Ensure the connection logic makes sense.
The core novelty is: (1) the agent knows when NOT to act, and (2) the cue tempo adapts to each person's own cadence, all running on the edge.
