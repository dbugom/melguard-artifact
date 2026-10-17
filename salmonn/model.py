"""MelGuard adaptation of the official SALMONN-13B inference model.

Derived from ``tsinghua-ee/SALMONN`` ``model.py`` (Apache-2.0, revision e0e903b8),
kept verbatim at ``salmonn/upstream/model.py``. Modifications, all mandated by the
locked spec (§5) — see ``salmonn/VENDORED.md``:

  1. Vicuna-13B backbone loaded in **4-bit NF4** (bitsandbytes, double-quant,
     fp16 compute) instead of the upstream fp16/int8 default. This is the
     deployed target ``f_S`` the attack differentiates through.
  2. Whisper + BEATs audio encoders kept in **fp16** (not quantised) so that
     input-gradients w.r.t. the waveform stay clean for the Phase 5–6 attack.
  3. Explicit device placement compatible with a quantised base (the whole
     module cannot be ``.to()``-moved once Vicuna is 4-bit), instead of the
     upstream ``model.to(device)`` in ``cli_inference.py``.
  4. A deterministic **greedy** generate path for the reproducibility gate.

Architecture, frame-splitting, Q-Former wiring and prompt handling are unchanged
from upstream. Component paths + all hyper-parameters are read from the lock by
``salmonn/loader.py``; nothing model-defining is hardcoded here.
"""
from __future__ import annotations

import os
import sys

# The vendored upstream uses absolute imports (`from beats...`, `from qformer...`).
_UPSTREAM = os.path.join(os.path.dirname(__file__), "upstream")
if os.path.isdir(_UPSTREAM) and _UPSTREAM not in sys.path:
    sys.path.insert(0, _UPSTREAM)

import torch

# peft >= 0.16 (we run 0.19.1) guards a LoRA path with
# `isinstance(w, torch.distributed.tensor.DTensor)`, but torch 2.5.1 does not
# auto-import `torch.distributed.tensor` into the `torch.distributed` namespace,
# so the attribute access raises AttributeError before our model is even built.
# Importing the submodule populates it (returns False for our bnb Params4bit).
try:  # noqa: SIM105
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
)

from beats.BEATs import BEATsConfig, BEATs  # type: ignore  # noqa: E402  (vendored, via sys.path)
from qformer.Qformer import BertConfig, BertLMHeadModel  # type: ignore  # noqa: E402


class MelGuardSALMONN(nn.Module):
    """SALMONN-13B (f_S) with a 4-bit NF4 Vicuna backbone and fp16 encoders."""

    def __init__(
        self,
        ckpt: str,
        whisper_path: str,
        beats_path: str,
        vicuna_path: str,
        *,
        speech_qformer_token_num: int = 1,
        speech_qformer_layer: int = 2,
        lora: bool = True,
        lora_alpha: int = 32,
        lora_rank: int = 8,
        lora_dropout: float = 0.1,
        second_per_frame: float = 0.333333,
        second_stride: float = 0.333333,
        quantization: str = "nf4",       # {"nf4", "int8", "fp16"} — spec §5 = nf4
        encoder_dtype: torch.dtype = torch.float16,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device_str = device
        self.encoder_dtype = encoder_dtype
        self._device_index = torch.device(device).index or 0

        # ---- feature extractor (numpy-domain, stays on CPU) ----
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)

        # ---- Whisper encoder (fp16) ----
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)

        # ---- BEATs audio encoder (fp16, frozen) ----
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location="cpu")
        beats_cfg = BEATsConfig(beats_checkpoint["cfg"])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint["model"])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        for _, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # ---- window-level speech Q-Former ----
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride

        # ---- Vicuna-13B backbone: 4-bit NF4 per spec §5 ----
        self.quantization = quantization
        self.llama_model = self._load_vicuna(vicuna_path, quantization)

        # ---- LoRA adaptor ----
        self.lora = lora
        if lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=None,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # ---- tokenizer ----
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"

        # ---- projection Q-Former -> LLM embedding space ----
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        # ---- load the trained SALMONN delta (Q-Former + proj + LoRA + LNs) ----
        ckpt_dict = torch.load(ckpt, map_location="cpu")["model"]
        missing, unexpected = self.load_state_dict(ckpt_dict, strict=False)
        # The Vicuna base weights are (correctly) "missing" — they come from vicuna_path,
        # not from salmonn_v1.pth. There should be no *unexpected* keys.
        self._ckpt_missing = len(missing)
        self._ckpt_unexpected = list(unexpected)

        # ---- place the non-quantised audio path on-device in fp16 ----
        self._place_audio_path(device, encoder_dtype)
        self.eval()

    # ------------------------------------------------------------------ helpers
    def _load_vicuna(self, vicuna_path: str, quantization: str):
        if quantization == "nf4":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            return LlamaForCausalLM.from_pretrained(
                vicuna_path,
                quantization_config=bnb,
                torch_dtype=torch.float16,
                device_map={"": self._device_index},
            )
        if quantization == "int8":
            return LlamaForCausalLM.from_pretrained(
                vicuna_path, load_in_8bit=True, torch_dtype=torch.float16,
                device_map={"": self._device_index},
            )
        # fp16 (matches upstream low_resource=False; not the deployed target)
        return LlamaForCausalLM.from_pretrained(
            vicuna_path, torch_dtype=torch.float16, device_map={"": self._device_index},
        )

    def _place_audio_path(self, device: str, dtype: torch.dtype) -> None:
        # Whisper encoder + LayerNorms + Q-Former + projection run in `dtype` (fp16).
        for module in (
            self.speech_encoder, self.ln_speech, self.ln_audio,
            self.speech_Qformer, self.speech_llama_proj,
        ):
            module.to(device=device, dtype=dtype)
        self.speech_query_tokens.data = self.speech_query_tokens.data.to(device=device, dtype=dtype)
        # BEATs computes its filterbank internally in fp32 (torchaudio kaldi
        # compliance), so its conv/patch-embedding weights must stay fp32 or the
        # fbank/weight dtypes mismatch. Its output is cast to `dtype` in
        # encode_audio; fp32 here also yields cleaner input-gradients for the attack.
        self.beats.to(device=device, dtype=torch.float32)

    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def encode_audio(self, wav, device=None):
        """Waveform (np.ndarray @16 kHz mono) -> speech embeds in the LLM space.

        This is the differentiable audio front-end (Whisper+BEATs -> Q-Former ->
        proj) the Phase 5–6 attack targets. Returns (B, L, llm_hidden)."""
        device = device or self.device_str
        dt = self.encoder_dtype
        spectrogram = self.feature_extractor(
            wav, return_tensors="pt", sampling_rate=16000
        ).input_features.to(device=device, dtype=dt)
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

        # BEATs runs in fp32 (its internal fbank is fp32); cast the output to `dt`.
        raw_wav = torch.from_numpy(wav).to(device=device, dtype=torch.float32).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(
            raw_wav, padding_mask=audio_padding_mask, feature_only=True
        )
        audio_embeds = audio_embeds.to(dt)

        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # window-level frame splitting (unchanged from upstream)
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, Ln = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], Ln)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        return speech_embeds

    @torch.inference_mode()
    def generate(
        self,
        wav_path: str,
        prompt: str,
        prompt_pattern: str = "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device: str | None = None,
        deterministic: bool = True,
        max_new_tokens: int = 200,
        num_beams: int = 1,
        min_length: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        top_p: float = 0.9,
        temperature: float = 1.0,
    ) -> list[str]:
        device = device or self.device_str

        # read + normalise wav (16 kHz mono, <=30 s)
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")

        speech_embeds = self.encode_audio(wav, device=device)

        embed_tokens = (
            self.llama_model.model.model.embed_tokens if self.lora
            else self.llama_model.model.embed_tokens
        )
        prompt_left, prompt_right = prompt_pattern.format(prompt).split("<SpeechHere>")
        left_ids = self.llama_tokenizer(prompt_left, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        right_ids = self.llama_tokenizer(prompt_right, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        left_embeds = embed_tokens(left_ids)
        right_embeds = embed_tokens(right_ids)
        bos = torch.ones([1, 1], dtype=torch.long, device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = embed_tokens(bos)

        embeds = torch.cat([bos_embeds, left_embeds, speech_embeds, right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long, device=device)

        gen_kwargs = dict(
            inputs_embeds=embeds,
            attention_mask=atts,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id,
        )
        if deterministic:
            gen_kwargs.update(do_sample=False, num_beams=1)  # greedy — fully reproducible
        else:
            gen_kwargs.update(do_sample=True, num_beams=num_beams, top_p=top_p, temperature=temperature)

        output = self.llama_model.generate(**gen_kwargs)
        return self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)
