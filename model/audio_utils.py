# Utils
# from glob import glob
from pathlib import Path

# Numbers
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import librosa.display

# Machine learning
import torch
import torchaudio
from torchaudio.functional import resample

# Audio
import librosa


def get_waveform(
    path,
    target_sr,
    duration=None,
    trim_silence=False,
    remove_all_silence=False,
    silence_top_db=35,
    silence_frame_length=2048,
    silence_hop_length=512,
):
    # Load waveform using torchaudio, if sample rate is different from target_sr, resample
    waveform, original_sr = torchaudio.load(path)
    if original_sr != target_sr:
        print(f"Resampling from {original_sr} to {target_sr}")
        waveform = resample(waveform, original_sr, target_sr)
    if waveform.ndim > 1:
        waveform = waveform[0, :]  # Use left channel

    waveform_np = waveform.numpy()

    if remove_all_silence:
        intervals = librosa.effects.split(
            waveform_np,
            top_db=silence_top_db,
            frame_length=silence_frame_length,
            hop_length=silence_hop_length,
        )
        if len(intervals) > 0:
            waveform_np = np.concatenate(
                [waveform_np[start:end] for start, end in intervals], axis=0
            )
        elif trim_silence:
            waveform_np, _ = librosa.effects.trim(
                waveform_np,
                top_db=silence_top_db,
                frame_length=silence_frame_length,
                hop_length=silence_hop_length,
            )
    elif trim_silence:
        waveform_np, _ = librosa.effects.trim(
            waveform_np,
            top_db=silence_top_db,
            frame_length=silence_frame_length,
            hop_length=silence_hop_length,
        )

    waveform = torch.from_numpy(waveform_np)

    if duration is not None:
        # Trim to custom duration
        waveform = waveform[: int(target_sr * duration)]
    return waveform


def get_specgram(waveform, win_length, hop_length, spec_in_db=True):
    F = torch.stft(
        waveform,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=True,
        window=torch.hann_window(win_length),
    ).T
    if spec_in_db:
        S = 10 * torch.log10(torch.abs(F) ** 2)
    else:
        S = torch.abs(F)
    return torch.angle(F), S


def get_spectrograms_from_audios(
    audio_path_list,
    target_sr,
    win_length,
    hop_length,
    db_min_norm=None,
    spec_in_db=True,
    normalize_each_audio=False,
    trim_silence=False,
    remove_all_silence=False,
    silence_top_db=35,
    silence_frame_length=2048,
    silence_hop_length=512,
):
    X = []
    y = []
    phases = []
    for i, filename in enumerate(audio_path_list):
        waveform = get_waveform(
            filename,
            target_sr,
            trim_silence=trim_silence,
            remove_all_silence=remove_all_silence,
            silence_top_db=silence_top_db,
            silence_frame_length=silence_frame_length,
            silence_hop_length=silence_hop_length,
        )
        phase, S = get_specgram(waveform, win_length, hop_length, spec_in_db=spec_in_db)
        if normalize_each_audio:
            S = S / S.max()
        phases.append(phase)
        X.append(S)
        y.append(torch.ones(S.shape[0]) * i)
    phases = torch.vstack(phases)
    X = torch.vstack(X)
    if spec_in_db and db_min_norm is not None:
        X = X.clip(db_min_norm, None) - db_min_norm
    X_max = float(X.max().numpy())
    X = X / X_max
    y = torch.hstack(y)
    return X, phases, X_max, y


def save_specgram(specgram, hop_length, path):
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(
        specgram.detach().numpy().T,
        y_axis="linear",
        x_axis="time",
        hop_length=hop_length,
    )
    plt.colorbar()
    plt.savefig(Path(path, "predicted_spectrogram.png"))


def save_latentscore(Z, hop_length, sr, path):
    plt.figure(figsize=(14, 4))
    t = np.arange(0, Z.shape[0]) * hop_length / sr
    plt.plot(t, Z + np.arange(Z.shape[1]), color="k")
    plt.savefig(path)


def spectrogram2audio(
    Y, db_min_norm, phase, hop_length, win_length, in_db, griffinlim=False, pghi=False
):
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    Y = torch.clamp(Y, min=0.0)

    if in_db:
        magnitude = torch.pow(10.0, (Y + db_min_norm) / 20.0)
    else:
        magnitude = Y

    magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = torch.clamp(magnitude, min=0.0)

    if pghi:
        from tifresi.stft import GaussTF
        stft_system = GaussTF(hop_size=hop_length, stft_channels=win_length)
        mag_np = magnitude.detach().cpu().numpy().T  # [freq_bins, frames]
        n_frames = mag_np.shape[1]
        # ltfatpy requires n_frames * hop_length divisible by win_length
        frames_per_window = win_length // hop_length
        remainder = n_frames % frames_per_window
        if remainder != 0:
            pad_frames = frames_per_window - remainder
            mag_np = np.pad(mag_np, ((0, 0), (0, pad_frames)), mode="constant")
        raw_audio = stft_system.invert_spectrogram(mag_np)
        # Trim padding
        expected_len = n_frames * hop_length
        audio = torch.tensor(raw_audio[:expected_len])
    elif griffinlim:
        audio = torch.tensor(
            librosa.griffinlim(
                magnitude.detach().cpu().numpy().T,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
            )
        )
    else:
        phase = torch.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
        Y_ = magnitude * torch.exp(1j * phase)
        audio = torch.istft(
            Y_.T,
            hop_length=hop_length,
            n_fft=win_length,
            window=torch.hann_window(win_length).to(Y_.device),
        )

    return torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def save_audio(Y, db_min_norm, phase, hop_length, win_length, samplerate, path, in_db):
    audio = spectrogram2audio(
        Y, db_min_norm, phase, hop_length, win_length, in_db, griffinlim=False
    )
    output_path = Path(path)

    save_kwargs = {}
    if output_path.suffix.lower() == ".mp3":
        save_kwargs["format"] = "mp3"
        save_kwargs["compression"] = torchaudio.io.CodecConfig(bit_rate=320)

    torchaudio.save(
        output_path,
        audio.reshape(1, -1),
        samplerate,
        **save_kwargs,
    )
