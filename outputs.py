from __future__ import annotations

import json
from pathlib import Path


def srt_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}時間{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def write_outputs(
    audio_path: Path,
    output_dir: Path,
    segments: list[dict],
    metadata: dict
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    plain_text = "".join(seg["text"].strip() for seg in segments).strip()
    timestamped_text = "\n".join(
        f"[{srt_timestamp(seg['start'])} - {srt_timestamp(seg['end'])}] {seg['text'].strip()}"
        for seg in segments
    )

    txt_path = output_dir / f"{stem}.txt"
    timestamped_path = output_dir / f"{stem}_timestamps.txt"
    srt_path = output_dir / f"{stem}.srt"
    vtt_path = output_dir / f"{stem}.vtt"
    json_path = output_dir / f"{stem}.json"

    txt_path.write_text(plain_text, encoding="utf-8-sig")
    timestamped_path.write_text(timestamped_text, encoding="utf-8-sig")

    srt_lines = []
    for idx, seg in enumerate(segments, 1):
        srt_lines.extend([
            str(idx),
            f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}",
            seg["text"].strip(),
            "",
        ])
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8-sig")

    vtt_lines = ["WEBVTT", ""]
    for seg in segments:
        vtt_lines.extend([
            f"{vtt_timestamp(seg['start'])} --> {vtt_timestamp(seg['end'])}",
            seg["text"].strip(),
            "",
        ])
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")

    payload = {
        "source_file": str(audio_path),
        **metadata,
        "text": plain_text,
        "segments": segments,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "TXT": txt_path,
        "タイムスタンプTXT": timestamped_path,
        "SRT": srt_path,
        "VTT": vtt_path,
        "JSON": json_path,
    }
