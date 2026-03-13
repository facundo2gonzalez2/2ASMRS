import csv
import math
import random
from pathlib import Path

import torch.nn.functional as F
import torchaudio


SUPPORTED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SCRIPT_DIR = Path(__file__).resolve().parent

# Simple configuration
INPUT_DIR = SCRIPT_DIR / "data_instruments"
OUTPUT_DIR = SCRIPT_DIR / "data_test"
SEGMENT_SECONDS = 2.0
SAMPLES_PER_INSTRUMENT = 10
SEED = 42
CLEAN_OUTPUT = True


def list_instrument_dirs(input_dir: Path):
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def collect_audio_files(instrument_dir: Path):
    return sorted(
        path
        for path in instrument_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def get_eligible_files(audio_files, segment_seconds: float):
    eligible = []
    too_short = []
    for audio_file in audio_files:
        try:
            info = torchaudio.info(str(audio_file))
        except Exception as exc:
            print(f"[warn] Could not inspect {audio_file}: {exc}")
            continue

        frames_needed = math.ceil(segment_seconds * info.sample_rate)
        if info.num_frames >= frames_needed:
            eligible.append((audio_file, info, frames_needed))
        else:
            too_short.append((audio_file, info.num_frames / info.sample_rate))
    return eligible, too_short


def clean_output_dir(output_dir: Path):
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        if path.is_file() and (
            path.suffix.lower() in SUPPORTED_SUFFIXES or path.name == "manifest.csv"
        ):
            path.unlink()


def load_random_segment(audio_file: Path, info, frames_needed: int, rng: random.Random):
    max_offset = info.num_frames - frames_needed
    start_frame = rng.randint(0, max_offset) if max_offset > 0 else 0
    waveform, sample_rate = torchaudio.load(
        str(audio_file), frame_offset=start_frame, num_frames=frames_needed
    )

    if waveform.shape[1] < frames_needed:
        waveform = F.pad(waveform, (0, frames_needed - waveform.shape[1]))

    start_seconds = start_frame / sample_rate
    return waveform, sample_rate, start_seconds


def write_manifest(output_dir: Path, rows):
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_name",
                "source_file",
                "sample_rate",
                "start_seconds",
                "segment_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def sample_instrument(
    instrument_name: str,
    instrument_dir: Path,
    output_dir: Path,
    samples_per_instrument: int,
    segment_seconds: float,
    rng: random.Random,
    clean_output: bool,
):
    audio_files = collect_audio_files(instrument_dir)
    eligible, too_short = get_eligible_files(audio_files, segment_seconds)

    if not eligible:
        print(
            f"[warn] Skipping {instrument_name}: no files long enough for {segment_seconds}s"
        )
        if too_short:
            print(
                f"[warn] {instrument_name}: found {len(too_short)} files but all were too short"
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clean_output_dir(output_dir)

    rows = []
    for sample_index in range(samples_per_instrument):
        audio_file, info, frames_needed = rng.choice(eligible)
        waveform, sample_rate, start_seconds = load_random_segment(
            audio_file, info, frames_needed, rng
        )
        sample_name = f"{instrument_name}_{sample_index:02d}.wav"
        sample_path = output_dir / sample_name
        torchaudio.save(str(sample_path), waveform, sample_rate)
        rows.append(
            {
                "sample_name": sample_name,
                "source_file": str(audio_file.relative_to(instrument_dir.parent)),
                "sample_rate": sample_rate,
                "start_seconds": f"{start_seconds:.3f}",
                "segment_seconds": f"{segment_seconds:.3f}",
            }
        )

    write_manifest(output_dir, rows)
    print(
        f"[ok] {instrument_name}: wrote {samples_per_instrument} samples to {output_dir}"
    )


def main():
    if SEGMENT_SECONDS <= 0:
        raise ValueError("SEGMENT_SECONDS must be greater than 0")
    if SAMPLES_PER_INSTRUMENT <= 0:
        raise ValueError("SAMPLES_PER_INSTRUMENT must be greater than 0")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    rng = random.Random(SEED)
    instrument_dirs = list_instrument_dirs(INPUT_DIR)
    if not instrument_dirs:
        raise RuntimeError(f"No instrument directories found in {INPUT_DIR}")

    for instrument_dir in instrument_dirs:
        sample_instrument(
            instrument_name=instrument_dir.name,
            instrument_dir=instrument_dir,
            output_dir=OUTPUT_DIR / instrument_dir.name,
            samples_per_instrument=SAMPLES_PER_INSTRUMENT,
            segment_seconds=SEGMENT_SECONDS,
            rng=rng,
            clean_output=CLEAN_OUTPUT,
        )


if __name__ == "__main__":
    main()
