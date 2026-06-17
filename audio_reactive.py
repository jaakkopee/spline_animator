"""Audio-reactive timeline generation module.

Analyzes audio files and generates timeline parameters based on audio features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioFeatures:
    """Extracted audio features."""
    duration_seconds: float
    sample_rate: int
    energy: np.ndarray
    spectral_centroid: np.ndarray
    zero_crossing_rate: np.ndarray
    onset_times: np.ndarray
    onset_strengths: np.ndarray
    chroma: np.ndarray
    frame_times: np.ndarray


def load_and_analyze_audio(audio_path: Path | str) -> AudioFeatures:
    """Load audio file and extract reactive features.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
        
    Returns:
        AudioFeatures containing extracted audio data
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Compute STFT and mel-spectrogram
    S = np.abs(librosa.stft(y))  # Use magnitude, not complex
    S_db = librosa.power_to_db(S ** 2, ref=np.max)
    mel_S = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_S, ref=np.max)

    # Energy
    energy = np.sqrt(np.sum(S ** 2, axis=0))
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_centroid = (spectral_centroid - np.min(spectral_centroid)) / (
        np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-8
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-8)

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr
    )
    onset_frames = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10
    )
    onset_times_detected = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames]

    # Chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma = (chroma - np.min(chroma)) / (np.max(chroma) - np.min(chroma) + 1e-8)

    # Frame times for alignment
    frame_times = librosa.frames_to_time(np.arange(len(energy)), sr=sr)

    return AudioFeatures(
        duration_seconds=float(duration),
        sample_rate=sr,
        energy=energy,
        spectral_centroid=spectral_centroid,
        zero_crossing_rate=zcr,
        onset_times=onset_times_detected,
        onset_strengths=onset_strengths,
        chroma=chroma,
        frame_times=frame_times,
    )


def extract_beat_timeline(
    features: AudioFeatures, fps: int = 24
) -> tuple[list[float], list[float]]:
    """Extract timeline segment boundaries and their energy values from audio.
    
    Uses onset detection to identify segment boundaries.
    
    Args:
        features: AudioFeatures from analyze_audio
        fps: Target frames per second for timeline
        
    Returns:
        Tuple of (segment_times, segment_energies)
    """
    if len(features.onset_times) < 2:
        # Fallback: use regular beat intervals
        beat_duration = features.duration_seconds / max(1, int(features.duration_seconds))
        segment_times = np.arange(0, features.duration_seconds + beat_duration, beat_duration)
    else:
        segment_times = np.concatenate([[0], features.onset_times])
        if segment_times[-1] < features.duration_seconds:
            segment_times = np.concatenate([segment_times, [features.duration_seconds]])

    # Get energy value for each segment (average within segment)
    segment_energies = []
    for i in range(len(segment_times) - 1):
        start_time = segment_times[i]
        end_time = segment_times[i + 1]
        mask = (features.frame_times >= start_time) & (features.frame_times < end_time)
        if np.any(mask):
            segment_energy = np.mean(features.energy[mask])
        else:
            segment_energy = 0.5
        segment_energies.append(float(segment_energy))

    return list(segment_times[:-1]), segment_energies


def map_features_to_easing(energy: float, spectral_centroid: float, zcr: float) -> str:
    """Map audio features to easing function.
    
    Args:
        energy: Normalized energy [0, 1]
        spectral_centroid: Normalized spectral centroid [0, 1]
        zcr: Normalized zero crossing rate [0, 1]
        
    Returns:
        Easing function name
    """
    # High energy and high spectral content -> sharp easing
    if energy > 0.6 and spectral_centroid > 0.5:
        return "ease-out"
    # Moderate energy -> smooth easing
    elif energy > 0.35:
        return "ease-in-out"
    # Low energy -> linear
    elif energy > 0.15:
        return "smoothstep"
    # Very low energy -> slow easing
    else:
        return "ease-in"


def map_features_to_spline_tension(energy: float) -> float:
    """Map energy to spline tension parameter.
    
    Args:
        energy: Normalized energy [0, 1]
        
    Returns:
        Spline tension in [-1, 1]
    """
    # High energy = tight curves (high tension)
    # Low energy = loose curves (low tension)
    return np.clip(energy * 1.5 - 0.5, -1.0, 1.0)


def map_features_to_frames(
    energy: float, min_frames: int = 8, max_frames: int = 64
) -> int:
    """Map audio features to segment frame count.
    
    Args:
        energy: Normalized energy [0, 1]
        min_frames: Minimum frame count
        max_frames: Maximum frame count
        
    Returns:
        Frame count for segment
    """
    # Higher energy = more frames (slower, more detailed animation)
    return max(min_frames, min(max_frames, int(min_frames + (energy * (max_frames - min_frames)))))


def _allocate_segment_frames(weighted_frames: np.ndarray, target_total: int) -> list[int]:
    """Allocate integer frame counts that sum to target_total.

    At least one frame is assigned to each segment when possible.
    """
    if target_total < 1:
        raise ValueError("target_total must be >= 1")
    if weighted_frames.size == 0:
        return []

    segment_count = int(weighted_frames.size)
    if target_total < segment_count:
        raise ValueError("target_total is too small for one-frame-per-segment allocation")

    normalized = weighted_frames.astype(np.float64)
    normalized = np.maximum(normalized, 1e-6)
    normalized = normalized * (target_total / float(np.sum(normalized)))

    frames = np.floor(normalized).astype(np.int64)
    frames = np.maximum(frames, 1)

    total = int(np.sum(frames))
    if total < target_total:
        need = target_total - total
        deficits = normalized - frames
        order = np.argsort(-deficits)
        for idx in order[:need]:
            frames[idx] += 1
    elif total > target_total:
        over = total - target_total
        removable = np.where(frames > 1)[0]
        if removable.size == 0:
            raise ValueError("Cannot reduce frames while preserving one frame per segment")

        surplus = frames - normalized
        order = removable[np.argsort(-surplus[removable])]
        ptr = 0
        while over > 0:
            idx = int(order[ptr % len(order)])
            if frames[idx] > 1:
                frames[idx] -= 1
                over -= 1
            ptr += 1

    return [int(value) for value in frames]


def generate_timeline_from_audio(
    audio_path: Path | str,
    keyframe_paths: list[Path | str] | None = None,
    fps: int = 24,
    min_segment_frames: int = 8,
    max_segment_frames: int = 64,
) -> dict[str, Any]:
    """Generate a timeline JSON from an audio file.
    
    The timeline duration will match the audio duration. Segment boundaries
    are determined by onset detection. Segment parameters are driven by
    audio features (energy, spectral content, etc.).
    
    If keyframe_paths are provided but there are fewer than needed, additional
    keyframes will be generated that cycle through the provided paths.
    
    Args:
        audio_path: Path to audio file
        keyframe_paths: List of keyframe image paths. If None, generates placeholder names.
        fps: Target frames per second
        min_segment_frames: Minimum frames per segment
        max_segment_frames: Maximum frames per segment
        
    Returns:
        Timeline dictionary ready for JSON serialization
    """
    features = load_and_analyze_audio(audio_path)

    # Extract segment boundaries from audio
    segment_times, segment_energies = extract_beat_timeline(features, fps=fps)

    target_total_frames = max(2, int(round(features.duration_seconds * fps)))
    target_segment_frames = target_total_frames - 1

    # Ensure one frame can be assigned per segment while preserving total duration.
    if len(segment_times) > target_segment_frames:
        downsample_count = target_segment_frames
        indices = [int(i * len(segment_times) / downsample_count) for i in range(downsample_count)]
        segment_times = [segment_times[idx] for idx in indices]
        segment_energies = [segment_energies[idx] for idx in indices]

    segment_end_times = segment_times[1:] + [features.duration_seconds]
    segment_durations = np.array(
        [max(1e-6, end - start) for start, end in zip(segment_times, segment_end_times)],
        dtype=np.float64,
    )

    # Use energy as a pacing weight while keeping the global frame budget tied to audio duration.
    reactive_preference = np.array(
        [map_features_to_frames(energy, min_segment_frames, max_segment_frames) for energy in segment_energies],
        dtype=np.float64,
    )
    preference_mean = float(np.mean(reactive_preference)) if reactive_preference.size else 1.0
    reactive_scale = reactive_preference / max(preference_mean, 1e-6)
    weighted_frames = segment_durations * fps * reactive_scale
    frames_per_segment = _allocate_segment_frames(weighted_frames, target_segment_frames)

    # Get audio features at key points
    segment_specs = []
    for i, (start_time, end_time, energy) in enumerate(
        zip(segment_times, segment_end_times, segment_energies)
    ):

        # Get average features for this segment
        mask = (features.frame_times >= start_time) & (features.frame_times < end_time)
        if np.any(mask):
            avg_spectral_centroid = np.mean(features.spectral_centroid[mask])
            avg_zcr = np.mean(features.zero_crossing_rate[mask])
        else:
            avg_spectral_centroid = 0.5
            avg_zcr = 0.5

        # Map features to segment parameters
        easing = map_features_to_easing(energy, avg_spectral_centroid, avg_zcr)
        frames = frames_per_segment[i]
        tension = map_features_to_spline_tension(energy)

        segment_specs.append({
            "frames": frames,
            "easing": easing,
            "interpolation": "catmull-rom",
            "alpha_blend": "premultiplied",
            "spline_tension": float(np.clip(tension, -1.0, 1.0)),
            "spline_bias": 0.0,
            "spline_continuity": float(np.clip(-0.15 if energy > 0.5 else 0.0, -1.0, 1.0)),
            "spline_endpoint": "mirror" if energy > 0.6 else "clamp",
        })

    # Generate keyframes: timeline requires num_keyframes = num_segments + 1
    num_keyframes_needed = len(segment_specs) + 1
    
    if keyframe_paths is None:
        # Generate placeholder keyframes
        keyframe_list = [f"frame_{i:03d}.png" for i in range(num_keyframes_needed)]
    else:
        # User provided keyframes
        provided_paths = [str(Path(p).name) for p in keyframe_paths]
        
        if len(provided_paths) >= num_keyframes_needed:
            # Enough keyframes provided, use them
            keyframe_list = provided_paths[:num_keyframes_needed]
        else:
            # Not enough keyframes - cycle through the provided ones to fill the gaps
            keyframe_list = []
            for i in range(num_keyframes_needed):
                idx = i % len(provided_paths)
                keyframe_list.append(provided_paths[idx])

    # Build timeline (schema-compliant, no extra fields)
    timeline: dict[str, Any] = {
        "keyframes": keyframe_list,
        "defaults": {
            "easing": "ease-in-out",
            "interpolation": "catmull-rom",
            "alpha_blend": "premultiplied",
            "spline_tension": 0.1,
            "spline_bias": 0.0,
            "spline_continuity": -0.15,
            "spline_endpoint": "mirror",
        },
        "segments": segment_specs,
    }

    return timeline


def print_audio_analysis(features: AudioFeatures) -> None:
    """Print audio analysis results for debugging."""
    print(f"Audio Analysis Report:")
    print(f"  Duration: {features.duration_seconds:.2f}s")
    print(f"  Sample Rate: {features.sample_rate} Hz")
    print(f"  Energy range: [{np.min(features.energy):.3f}, {np.max(features.energy):.3f}]")
    print(f"  Spectral Centroid range: [{np.min(features.spectral_centroid):.3f}, {np.max(features.spectral_centroid):.3f}]")
    print(f"  Zero Crossing Rate range: [{np.min(features.zero_crossing_rate):.3f}, {np.max(features.zero_crossing_rate):.3f}]")
    print(f"  Detected Onsets: {len(features.onset_times)}")
    if len(features.onset_times) > 0:
        print(f"    Times: {features.onset_times[:min(10, len(features.onset_times))]}")
