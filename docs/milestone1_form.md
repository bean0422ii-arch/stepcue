# Milestone 1: Project Registration Form (작성용 문구)

> `IoT_Project_Registration_Form.docx`의 각 칸에 그대로 복사해서 넣는다.
> `[ ]` 칸(이름, 학번, 이메일, 서명, 날짜)은 직접 채운다. 개인 정보는 이 파일에 적지 않는다.
> Word에 붙여넣을 때 깨지지 않도록 본문에는 백틱(`)을 쓰지 않는다. 한국어 확인용: [milestone1_form_ko.md](milestone1_form_ko.md)

---

## 1. Project Nature
☑ Solo (1 student)

## 2. Team
**Team name:** StepCue

| No. | Full name | Student ID | Email | Role (A / B) | Exchange? (Y/N) |
|---|---|---|---|---|---|
| 1 | [직접] | [직접] | [직접] | A | [직접 Y/N] |
| 2 | – | – | – | – | – |

## 3. Project selection
**Project ID:** (Professor will assign, leave blank)

**Project title:** StepCue – A Two-Layer Edge AI Agent for Personalized Freezing-of-Gait Cueing in Parkinson's Disease

## 4. Problem statement
**Who is the user?**
A person with Parkinson's disease who walks alone at home (e.g., a 72-year-old who is often alone at home during the day), and a caregiver who should be told when something serious happens.

**What is the problem?**
People with Parkinson's disease often experience freezing of gait (FoG): while walking, their feet suddenly feel stuck to the floor, which can cause falls. A rhythmic cue (metronome beat or vibration) helps them start walking again, but only if it comes within about 1 second and matches their own walking pace. A cue given when the person stops on purpose is just a disturbance, and a device that gives too many false cues will not be used.

**What will your system detect?**
From body-worn accelerometer data (2-second windows, updated every 0.5 s):
- fog_onset: freezing starts while the person was walking
- fog_prolonged: freezing lasts more than 3 s (cue not working) or more than 10 s (fall risk)
- fog_recovered: normal walking resumes
- intentional_stop: the person stopped on purpose (no rhythmic walking before) → no action
- walking_cadence: the person's own steps per minute, used as the cue tempo

**What will the agent do?**
The agent has two layers:
1. **Fast edge layer (rule-based, < 1 s, works offline):** starts a metronome cue at the person's own cadence, slows it to 90% and adds vibration if freezing continues for 3 s, stops the cue on recovery, notifies the caregiver if freezing lasts over 10 s or happens 3+ times in 10 minutes, and does nothing when the stop looks intentional. Every decision is logged with its inputs and reason.
2. **Slow LLM layer (Claude, outside the real-time loop):** after a session, it reads a summary of the decision log (never raw sensor data), uses tools to re-run the detector on recorded data with different thresholds, and personalizes the detection threshold to reduce false cues for that user.
3. **Learning from feedback:** the dashboard has a "Wrong cue" button. Each press is logged with the context (FoG probability, walking rhythm, cadence). The LLM agent reads this feedback, finds common patterns (e.g., false cues right after turning), saves them as a per-user memory, and proposes new settings. The effect of every change is recorded (before/after false cues per 10 minutes).

**When it asks a human first:** small threshold changes within a safe range (0.6–0.9) are applied automatically with a written reason; larger changes, or changes outside the range, are sent to the user/caregiver for approval on the dashboard.

## 5. Data to be used
**Dataset name(s):** Daphnet Freezing of Gait Dataset (10 subjects with Parkinson's disease, 3 accelerometers on ankle, thigh and hip, 64 Hz, 237 annotated freezing episodes). Optional: my own smartphone accelerometer recordings of normal walking, used only as a "no-freezing" test.

**Source / link:** https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait
(Project repository: https://github.com/bean0422ii-arch/stepcue)

**Type of data:** ☑ Open dataset

**Limitation:** The data is pre-recorded, so the people in it do not react to our cues. Whether we play a cue or change its tempo, the recorded movement stays the same. This means the project cannot test whether a cue actually shortens a freezing episode, or which tempo works better. Instead, we evaluate the parts the system decides on its own: reaction time, detection rate, false cues per 10 minutes, and whether the cue tempo matches the person's cadence. The "Wrong cue" feedback is used to reduce unnecessary cues; it does not measure how well a cue helps. Testing the real effect (e.g., logging the time from cue start to walking again) is left as future work with a wearable device and real users.

## 6. AI tools to be used
**Tools for building:** ☑ Claude (Claude Code)

**LLM inside your agent:** Anthropic Claude API (model: Claude Opus 5.5, claude-opus-5-5), called with tool use only by the slow calibration layer. The real-time cueing runs locally without any LLM, so the system keeps working if the network is down.

**What AI will help with:** ☑ Writing code ☑ Debugging ☑ Agent prompts and design ☑ Slides and report
(Synthetic data: not used)

## 7. Declaration
☑ We will not put real personal or health data into any AI tool.
☑ Every member will contribute and be able to explain the code.

| Member 1 signature | Member 2 signature | Date |
|---|---|---|
| [직접 서명] | – | [제출일] |
