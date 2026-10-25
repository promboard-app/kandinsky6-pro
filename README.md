# Kandinsky 6.0 Pro — 소리까지 든 영상 만들기 (29B)

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **Kandinsky 6 Pro 영상+소리** 노드 하나를 더합니다. Kandinsky Lab 이 공개한 Kandinsky 6.0 Pro(29B)로 프롬프트에서, 또는 시작 프레임 그림에서 이어지는 5초 영상을 소리(44.1kHz)까지 함께 만듭니다. 같은 팀의 Lite(2B) 보다 화질 · 움직임이 낫고 훨씬 무겁습니다.

> 🔴 **미검증**: 아직 실제로 돌려 보지 않았습니다. 이 플러그인을 만든 PC(그래픽카드 12GB)는 공식 최소 사양(16GB)에 못 미칩니다. 문제가 있으면 [이슈](https://github.com/promboard-app/kandinsky6-pro/issues)로 알려 주세요.

## 입력과 결과

| 포트 | 종류 | 뜻 |
|---|---|---|
| 프롬프트 | 글(필수) | 장면 · 움직임 · 카메라를 영어(또는 러시아어)로 자세히. `Audio: rain, distant thunder` 처럼 소리를 적으면 그 소리를 만듭니다 |
| 시작 프레임 | 그림(선택) | 이으면 그 그림에서 시작하는 영상(가운데를 맞춰 잘립니다) |
| 영상 | 영상 | 5초 mp4(H.264, 24fps) + 소리(AAC 44.1kHz) |
| 소리 | 소리 | 영상의 소리만 담은 wav |

- 설정: 모델(Pro 빠른 판 · Pro 일반 판), 해상도(SD 864×480 · HD 1280×720), 화면 방향(시작 프레임이 없을 때), 소리(넣기 · 빼기), 시드, 피할 것(일반 판만).
- 생성 값은 공식 값입니다: 빠른 판 10단계 · guidance 1.0(PiFlow 스케줄러), 일반 판 50단계 · guidance 5.0, 121프레임(5초).
- 프롬프트는 영어 · 러시아어로 학습됐습니다. 한국어로 적으면 결과가 덜 맞을 수 있습니다.

## 받는 것과 용량 (고른 판의 변환기만, 처음 실행할 때 한 번)

| 파일 | 출처 | 크기 |
|---|---|---|
| 글 인코더 Qwen2.5-VL-7B · CLIP-L, VAE, 소리 VAE, 보코더 | [kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers) (Pro 저장소와 같은 파일 — Kandinsky 6 Lite 플러그인과 같이 씁니다) | 20.3GB |
| Pro 빠른 판 변환기 | [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) | 60.3GB |
| Pro 일반 판 변환기 | [kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers) (허깅페이스 사용 승인 필요) | 69.7GB |

합계: 빠른 판 약 81GB, 일반 판 약 90GB.

## 필요한 사양

- NVIDIA 그래픽카드 전용(CUDA), **16GB 이상**(공식 16GB 프리셋). 변환기 · VAE 를 블록 단위로 올렸다 내리며 돕니다.
- RAM **64GB 이상**. 공식 16GB 프리셋이 호스트 RAM 을 약 58GiB 씁니다(Kandinsky 6.0 기술 보고서).
- PromBoard 는 노드 한 번 실행을 약 15분으로 끊습니다. 그 안에 끝나지 않을 것 같으면 노드가 먼저 멈추고 해상도 · 판을 바꾸라고 알립니다.

## 라이선스와 조건

- 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).
- 모델 가중치: MIT(Kandinsky 6.0). 매출 상한 · 지역 제한이 없습니다. Pro 일반 판 저장소만 허깅페이스에서 동의 버튼을 눌러야 받을 수 있습니다(자동 승인).

## 설치

1. PromBoard 앱 → 플러그인 매니저 → **커뮤니티 플러그인** → 둘러보기에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. Pro 일반 판을 쓸 때만: 허깅페이스에 가입하고 [모델 페이지](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers)에서 동의한 뒤, Read 토큰을 만들어 플러그인 설정의 `HF_TOKEN` 칸에 붙여 넣습니다.
4. 처음 실행할 때 실행 준비가 파이썬 패키지와 고른 판의 모델을 받습니다.

## 플러그인 만드는 사람을 위한 메모

```
manifest.json            플러그인 정보 · 노드 정의 · 모델(requires.models)
nodes/main.py            NODES = {"kandinsky6": 함수} — 글 인코더로 임베딩을 만든 뒤 Kandinsky6TI2VAPipeline 으로 영상+소리 생성
nodes/requirements*.txt  받을 패키지 · 운영체제별 해시 잠금 파일(Windows · Linux)
```

- diffusers 0.42.0 의 `Kandinsky6TI2VAPipeline` 을 씁니다. 글 인코더는 파이프라인에 넣지 않고 따로 임베딩을 만들어 넘깁니다(Qwen 을 블록 단위로 올렸다 내리려고).
- 잠금 파일 다시 만들기: `uv pip compile nodes/requirements.in --generate-hashes --python-version 3.12 --python-platform x86_64-pc-windows-msvc --index-url https://pypi.org/simple --exclude-newer 2026-10-25T00:00:00Z` (리눅스는 `x86_64-manylinux_2_28`). torch 줄은 앱이 빼고 앱의 torch 2.8.0 을 먼저 깝니다(`"torch": "host"`).

---

## English

A PromBoard community plugin. It adds one **Kandinsky 6 Pro video+audio** node that runs Kandinsky Lab's open Kandinsky 6.0 Pro (29B): a 5-second video with matching audio (44.1 kHz), from a prompt or continuing from a start frame. Better quality and motion than Lite (2B), and much heavier.

> 🔴 **Unverified**: not run yet. The 12 GB card this plugin was built on is below the official minimum (16 GB). Please report problems in [issues](https://github.com/promboard-app/kandinsky6-pro/issues).

- Inputs: prompt (required; English or Russian works best — write `Audio: …` to describe the sound), start frame (optional, center-cropped). Outputs: a 5 s MP4 (H.264 24 fps + AAC 44.1 kHz) and the audio alone as WAV.
- Settings: model (Pro fast = official distilled, 10 steps · guidance 1.0 · PiFlow; Pro normal = 50 steps · guidance 5.0), resolution (SD 864×480 · HD 1280×720), orientation (without a start frame), sound on/off, seed, negative prompt (normal only).
- Downloads (only the chosen transformer, once): shared text encoders (Qwen2.5-VL-7B · CLIP-L) · VAE · audio VAE · vocoder from kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers (same files as the Pro repos, shared with the Kandinsky 6 Lite plugin, 20.3 GB) + Pro fast transformer (60.3 GB) or Pro normal transformer (69.7 GB, Hugging Face access approval needed). About 81 GB / 90 GB in total.
- Requirements: NVIDIA GPU (CUDA), **16 GB or more** (official 16 GB preset; the transformer and VAE are block-offloaded). **64 GB+ RAM** — the official 16 GB preset uses about 58 GiB of host RAM. PromBoard stops one node run after about 15 minutes; the node stops earlier and says so when it will not make it.
- License: plugin code Apache-2.0; model weights MIT, no revenue cap or regional restriction. Only the Pro normal repo needs a click-through approval on Hugging Face (automatic).
- **Install**: PromBoard → Plugin Manager → **Community plugins** → Browse → install. Turn off **Restricted mode** and enable the plugin. For Pro normal only, accept the terms on the [model page](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers) and paste a Read token into the plugin's `HF_TOKEN` field.

The plugin texts in the app are Korean only for now — the plugin manifest takes a single language.

## License

Plugin code: Apache-2.0 — see [LICENSE](LICENSE). The Kandinsky 6.0 weights are MIT and are downloaded from Hugging Face, not bundled.
