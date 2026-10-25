"""promboard.kandinsky6-pro — Kandinsky 6.0 Pro(29B) 영상+소리 만들기 노드 코드. 🔴 검증 안 됨: 이 PC(12GB)에서는 돌려 보지 않았다.

노드 하나(kandinsky6)가 프롬프트(필수)와 시작 프레임(선택)을 받아 소리가 든 5초 mp4(H.264 + AAC 44.1kHz) 한 편과
그 소리만 따로 담은 wav 를 낸다. 시작 프레임을 이으면 그 그림에서 시작하는 영상(이미지→영상+소리)이다.

모델(설정 model, manifest requires.models 의 when 으로 고른 것만 받는다) — kandinskylab 공식 Diffusers 저장소
  distill  Kandinsky-6.0-Pro-distill-5s-Diffusers: PiFlow 스케줄러, 10단계, guidance 1.0(부정 프롬프트를 쓰지 않는다)
  base     Kandinsky-6.0-Pro-5s-Diffusers(허깅페이스 승인 필요): FlowMatch 스케줄러, 50단계, guidance 5.0
  Pro 와 Lite 는 글 인코더(Qwen2.5-VL-7B · CLIP-L) · VAE · 소리 VAE · 보코더 파일이 같다(sha256 같음) — 공통 파일은 Lite 빠른 판
  저장소에서 받아 Kandinsky 6 Lite 플러그인과 같이 쓴다.
공식 값(저장소 README · diffusers Kandinsky6TI2VAPipeline 예시): 480×864, 121프레임(5초), 24fps, 소리 44.1kHz.

메모리(공식 16GB 프리셋은 변환기 블록 2개만 그래픽카드에 두고 호스트 RAM 약 58GiB 를 쓴다 — 논문 §6)
  - 변환기(bf16 약 58GiB)는 블록 단위로 올렸다 내린다(diffusers group offloading, block_level). VAE · 소리 VAE · 보코더도 같은 방식.
  - Qwen2.5-VL-7B(bf16 약 16GB)는 그래픽카드에 통째로 못 올려 층(leaf) 단위로 올렸다 내리며 프롬프트를 임베딩으로 바꾼다
    (diffusers group offloading). 파이프라인에는 글 인코더를 넣지 않고 임베딩을 넘긴다.
    파이프라인은 프롬프트를 1024 토큰으로 채우고 채운 자리를 마스크로 가리는데, 임베딩을 넘기면 마스크를 받을 칸이 없어
    채운 자리를 잘라 낸 임베딩을 넘긴다(가린 토큰을 빼는 것과 같다 — 파이프라인도 채운 자리가 없으면 마스크를 None 으로 둔다).
  - 글 인코더(CLIP, 그래픽카드 20GB 이상이면 Qwen 통째로)는 인코딩 때만 그래픽카드에 올리고 끝나면 RAM 으로 내린다(_encode) —
    남겨 두면 블록씩 올리는 변환기 · VAE 차례에도 그래픽카드를 차지한다. Kandinsky 6 Lite 와 같다.
  - VAE 는 타일로 나눠 푼다.

노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import gc
import math
import os
import random
import time

COMMON = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
VARIANTS = {
    "distill": {"repo": "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", "steps": 10, "guidance": 1.0},
    "base": {"repo": "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers", "steps": 50, "guidance": 5.0},
}
SIZES = {"sd": (864, 480), "hd": (1280, 720)}  # (가로, 세로) 가로 영상 기준
NUM_FRAMES = 121  # 5초 · 24fps(5s 체크포인트)
FPS = 24
MAX_SEQUENCE_LENGTH = 1024  # 파이프라인 기본
TE_ON_GPU_MB = 20_000  # 이 이상이면 Qwen 을 통째로 올린다
NO_TILING_MB = 24_000
# PromBoard 실행기는 노드 한 번에 900초 + 처음 모델 적재 몫 180초를 준다(server/src/plugins/runner/run.ts). 넘을 것 같으면 멈추고 알린다.
TIME_BUDGET_SEC = 1050
DECODE_SHARE = 0.15

_CACHE = {"key": None, "pipe": None, "te": None}


def _repo_dir(ctx, repo, probe):
    """받은 저장소 폴더(<모델 폴더>/_plugin-models/<저장소>@<커밋>/). probe = 그 안의 받은 파일 하나."""
    path = os.path.normpath(ctx.model_path(f"{repo}/{probe}"))
    for _ in probe.split("/"):
        path = os.path.dirname(path)
    return path


def _vram_mb():
    import torch

    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if torch.cuda.is_available() else 0


def _free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Progress:
    def __init__(self, ctx, lo=0.0, hi=1.0):
        self.ctx, self.lo, self.hi, self.last = ctx, lo, hi, lo

    def __call__(self, fraction, stage="work"):
        value = max(self.last, self.lo + (self.hi - self.lo) * max(0.0, min(1.0, fraction)))
        self.last = value
        self.ctx.progress(value, stage)


def _text_encoders(ctx, progress):
    import torch
    from diffusers.hooks import apply_group_offloading
    from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    if _CACHE["te"] is not None:
        return _CACHE["te"]
    progress(0.0, "load-model")
    start = time.monotonic()
    root = _repo_dir(ctx, COMMON, "text_encoder/config.json")
    processor = Qwen2_5_VLProcessor.from_pretrained(os.path.join(root, "tokenizer"))
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(root, "text_encoder"), torch_dtype=torch.bfloat16)
    qwen.eval()
    tok2 = CLIPTokenizer.from_pretrained(os.path.join(root, "tokenizer_2"))
    clip = CLIPTextModel.from_pretrained(os.path.join(root, "text_encoder_2"), torch_dtype=torch.bfloat16)
    clip.eval()
    if _vram_mb() < TE_ON_GPU_MB:
        # 층(leaf) 단위로 올렸다 내린다. 블록 단위(block_level)는 바로 아래 자식의 ModuleList 만 묶는데 Qwen2.5-VL 은 층이
        # model.language_model.layers 로 깊이 있어, 모델 전체(15.4GiB)가 한 묶음으로 올라가 12GB 를 넘었다(실측 16.4GiB → 층 단위 1.9GiB, 결과 같음).
        apply_group_offloading(qwen, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="leaf_level")
    _CACHE["te"] = (processor, qwen, tok2, clip)
    ctx.log(f"text encoders loaded in {time.monotonic() - start:.1f}s")
    return _CACHE["te"]


def _encode(ctx, progress, texts):
    """글 목록 → [(Qwen 임베딩 (1, 실제 길이, 3584), CLIP 풀링 (1, 768))]. 채운 자리는 잘라 낸다(맨 위 설명)."""
    import torch
    from diffusers import Kandinsky6TI2VAPipeline

    processor, qwen, tok2, clip = _text_encoders(ctx, progress)
    progress(0.0, "encode")
    # 그래픽카드에는 쓸 때만 올린다(모델 읽기는 그래픽카드를 쓰지 않게). 층 단위로 내리지 않는 큰 그래픽카드는 Qwen 도 통째로.
    whole = _vram_mb() >= TE_ON_GPU_MB
    clip.to("cuda")
    if whole:
        qwen.to("cuda")
    out = []
    try:
        for text in texts:
            with torch.no_grad():
                emb, pooled, mask = Kandinsky6TI2VAPipeline._get_prompt_embeds(
                    [text], processor, qwen, tok2, clip, MAX_SEQUENCE_LENGTH, torch.device("cuda")
                )
            if mask is not None:
                emb = emb[:, mask[0]]
            out.append((emb.to(torch.bfloat16), pooled.to(torch.bfloat16)))
            ctx.check_cancel()
    finally:
        # 멈추거나 실패해도 내린다 — 변환기 · VAE 가 쓸 그래픽카드 메모리를 글 인코더가 쥐고 있지 않게.
        clip.to("cpu")
        if whole:
            qwen.to("cpu")
        _free_cuda()
    return out


def _pipeline(ctx, progress, variant):
    import torch
    from diffusers import (
        AutoencoderKLHunyuanVideo,
        FlowMatchEulerDiscreteScheduler,
        Kandinsky6TI2VAPipeline,
        Kandinsky6Transformer3DModel,
        MMAudioVAE,
        MMAudioVocoder,
        PiflowScheduler,
    )

    if _CACHE["key"] == variant:
        return _CACHE["pipe"]
    _CACHE.update(key=None, pipe=None)
    _free_cuda()
    progress(0.0, "load-model")
    start = time.monotonic()
    common = _repo_dir(ctx, COMMON, "vae/config.json")
    root = _repo_dir(ctx, VARIANTS[variant]["repo"], "transformer/config.json")
    transformer = Kandinsky6Transformer3DModel.from_pretrained(os.path.join(root, "transformer"), torch_dtype=torch.bfloat16)
    ctx.check_cancel()
    scheduler_cls = PiflowScheduler if variant == "distill" else FlowMatchEulerDiscreteScheduler
    scheduler = scheduler_cls.from_pretrained(os.path.join(root, "scheduler"))
    vae = AutoencoderKLHunyuanVideo.from_pretrained(os.path.join(common, "vae"), torch_dtype=torch.bfloat16)
    audio_vae = MMAudioVAE.from_pretrained(os.path.join(common, "audio_vae"), torch_dtype=torch.bfloat16)
    vocoder = MMAudioVocoder.from_pretrained(os.path.join(common, "vocoder"), torch_dtype=torch.bfloat16)
    pipe = Kandinsky6TI2VAPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        scheduler=scheduler,
        audio_vae=audio_vae,
        vocoder=vocoder,
    )
    # 변환기가 그래픽카드보다 크다 — 블록 단위로 올렸다 내린다(공식 16GB 프리셋과 같은 생각, 논문 §6).
    pipe.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="block_level", num_blocks_per_group=2)
    if _vram_mb() < NO_TILING_MB:
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    _CACHE.update(key=variant, pipe=pipe)
    ctx.log(f"pipeline {variant} loaded in {time.monotonic() - start:.1f}s")
    return pipe


def kandinsky6(ctx, inputs, settings):
    import numpy as np
    import torch
    from diffusers.utils import encode_video
    from PIL import Image, ImageOps

    t0 = time.monotonic()
    prompt = (inputs.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("프롬프트를 이어 주세요. 장면 · 움직임 · 카메라와 'Audio: …'로 소리까지 영어로 자세히 적을수록 잘 나옵니다.")
    if ctx.device != "cuda":
        raise ValueError("Kandinsky 6.0 Pro 는 NVIDIA 그래픽카드(CUDA)에서만 돕니다. 이 PC 에서는 그래픽카드를 찾지 못했습니다.")
    variant = settings["model"]
    image = None
    if inputs.get("image"):
        with Image.open(inputs["image"]) as im:
            im.load()
            image = ImageOps.exif_transpose(im).convert("RGB")
    w, h = SIZES[settings["resolution"]]
    portrait = (image.height > image.width) if image is not None else settings["orientation"] == "portrait"
    width, height = (h, w) if portrait else (w, h)
    sound = settings["sound"] == "on"
    steps = VARIANTS[variant]["steps"]
    guidance = VARIANTS[variant]["guidance"]
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    negative = (settings.get("negative") or "").strip()
    ctx.log(f"{variant} {width}x{height} {NUM_FRAMES}f {steps} steps seed {seed} image={image is not None} sound={sound}")

    try:
        load = Progress(ctx, 0.0, 0.1)
        texts = [prompt]
        if guidance > 1.0:
            from diffusers import Kandinsky6TI2VAPipeline

            texts.append(negative or Kandinsky6TI2VAPipeline._DEFAULT_NEGATIVE_PROMPT)
        embeds = _encode(ctx, load, texts)
        pipe = _pipeline(ctx, load, variant)
        ctx.check_cancel()

        denoise = Progress(ctx, 0.1, 0.9)
        loop_start = [None]

        def on_step(p, i, t, kwargs):
            ctx.check_cancel()
            now = time.monotonic()
            if loop_start[0] is None:
                loop_start[0] = now
            done = i + 1
            denoise(done / steps, "denoise")
            if done >= 2:
                per = (now - loop_start[0]) / max(1, done - 1)
                total = (now - t0) + per * (steps - done) * (1 + DECODE_SHARE) + 20
                if total > TIME_BUDGET_SEC:
                    raise TimeoutError(
                        f"이 PC 에서는 이 설정으로 약 {math.ceil(total / 60)}분이 걸려 한 번에 돌 수 있는 시간(약 {TIME_BUDGET_SEC // 60}분)을"
                        " 넘습니다. 해상도를 SD 로, 모델을 'Pro 빠른 판'으로 바꿔 다시 실행해 주세요."
                    )
            return kwargs

        call = dict(
            prompt_embeds=embeds[0][0],
            pooled_prompt_embeds=embeds[0][1],
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            frame_rate=float(FPS),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            sample_audio=sound,
            output_type="np",
            callback_on_step_end=on_step,
        )
        if guidance > 1.0:
            call["negative_prompt_embeds"] = embeds[1][0]
            call["negative_pooled_prompt_embeds"] = embeds[1][1]
        if image is not None:
            call["image"] = image
        torch.cuda.reset_peak_memory_stats()
        result = pipe(**call)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        ctx.check_cancel()
        ctx.progress(0.95, "encode")
        frames = result.frames[0]
        audio = None
        if sound and result.audio is not None:
            audio = torch.from_numpy(np.asarray(result.audio[0], dtype=np.float32)).reshape(1, -1)
        out = ctx.out_path("kandinsky6-pro.mp4")
        encode_video(frames, fps=FPS, output_path=out, audio=audio, audio_sample_rate=pipe.audio_sample_rate if audio is not None else None)
        outputs = {"video": out}
        if audio is not None:
            outputs["audio"] = _write_wav(ctx.out_path("kandinsky6-pro.wav"), audio[0].numpy(), pipe.audio_sample_rate)
        ctx.log(f"done in {time.monotonic() - t0:.1f}s, peak torch VRAM {peak:.2f} GiB")
    except torch.OutOfMemoryError:
        oom = True
    else:
        oom = False
    finally:
        _free_cuda()
    if oom:
        # except 블록 밖에서 정리하고 새 오류를 낸다 — 블록 안에서 내면 원래 오류(와 그 프레임이 쥔 그래픽카드 텐서)가
        # 새 오류의 __context__ 로 남아 사이드카가 다음 실행까지 메모리를 놓지 못한다.
        _CACHE.update(key=None, pipe=None)
        _free_cuda()
        raise RuntimeError(
            "그래픽카드 메모리가 모자라 멈췄습니다. 해상도를 SD 로 낮춰 주세요. 다른 프로그램(게임 · 다른 AI)이 그래픽카드를 쓰고 있으면 닫아 주세요."
        )
    ctx.progress(1.0, "done")
    return outputs


def _write_wav(path, samples, rate):
    """float -1~1 모노 → 16비트 PCM wav."""
    import wave

    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())
    return path


NODES = {"kandinsky6": kandinsky6}
