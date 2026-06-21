import glob
import os
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile

import yaml


def convert_mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int, channels: int) -> bool:
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    filename = os.path.basename(mp3_path)
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-ar", str(sample_rate), "-ac", str(channels), "-y", wav_path],
            capture_output=True,
        )
    except FileNotFoundError:
        print(f"FAILED: {filename}")
        return False
    if result.returncode == 0:
        print(f"Converted: {filename}")
        return True
    else:
        print(f"FAILED: {filename}")
        return False


def normalise_loudness(wav_path: str, target_lufs: float) -> bool:
    filename = os.path.basename(wav_path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(wav_path))
    os.close(tmp_fd)
    try:
        # Try ffmpeg-normalize first
        result = subprocess.run(
            ["ffmpeg-normalize", wav_path, "-o", tmp_path,
             "-t", str(target_lufs), "--true-peak", "-1.5", "--loudness-range-target", "11",
             "-ar", "16000", "-f"],
            capture_output=True,
        )
        used_fallback = False
    except FileNotFoundError:
        used_fallback = True
        result = None

    if used_fallback or (result is not None and result.returncode != 0):
        # Fall back to ffmpeg loudnorm filter
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", wav_path,
                 "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                 "-ar", "16000", "-ac", "1", "-y", tmp_path],
                capture_output=True,
            )
        except FileNotFoundError:
            os.remove(tmp_path)
            print(f"FAILED: {filename}")
            return False

    if result.returncode == 0:
        shutil.move(tmp_path, wav_path)
        print(f"Normalised: {filename}")
        return True
    else:
        os.remove(tmp_path)
        print(f"FAILED: {filename}")
        return False


def extract_readaloud_text(zip_path: str, ebible_num: str, code: str, chapter: int) -> "str | None":
    filename = f"suzBl_{ebible_num}_{code}_{chapter:02d}_read.txt"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                with zf.open(filename) as f:
                    return f.read().decode("utf-8").strip()
            except KeyError:
                return None
    except FileNotFoundError:
        return None


def _wav_duration_seconds(wav_path: str) -> float:
    try:
        with wave.open(wav_path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/audio.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mp3_dir        = cfg["mp3_dir"]
    readaloud_zip  = cfg["readaloud_zip"]
    wav_dir        = cfg["wav_dir"]
    pairs_dir      = cfg["pairs_dir"]
    sample_rate    = cfg["sample_rate"]
    channels       = cfg["channels"]
    loudness_target = cfg["loudness_target"]
    book_map       = cfg["book_map"]
    pilot_only     = cfg.get("pilot_only", False)
    pilot_book     = cfg.get("pilot_book", None)

    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)

    books = {pilot_book: book_map[pilot_book]} if pilot_only and pilot_book else book_map

    total = failed = succeeded = 0
    total_duration_s = 0.0

    for book_key, book in books.items():
        code    = book["code"]
        ebible  = book["ebible"]
        chapters = book["chapters"]

        for ch in range(1, chapters + 1):
            total += 1

            # Locate MP3 by glob — filename book-name field varies in length/spelling
            pattern = os.path.join(mp3_dir, f"{book_key}___{ch:02d}_*SUZWBTN1DA.mp3")
            matches = glob.glob(pattern)
            if not matches:
                print(f"MISSING MP3: {book_key} ch{ch:02d}")
                failed += 1
                continue
            mp3_path = matches[0]

            wav_path = os.path.join(wav_dir, f"{code}_{ch:03d}.wav")

            if not convert_mp3_to_wav(mp3_path, wav_path, sample_rate, channels):
                failed += 1
                continue

            normalise_loudness(wav_path, loudness_target)
            total_duration_s += _wav_duration_seconds(wav_path)

            text = extract_readaloud_text(readaloud_zip, ebible, code, ch)
            if text:
                txt_path = os.path.join(pairs_dir, f"{code}_{ch:03d}.txt")
                with open(txt_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            else:
                print(f"NO TRANSCRIPT: {code} ch{ch:02d}")

            succeeded += 1

    print()
    print("--- Summary ---")
    print(f"Total MP3s processed:   {total}")
    print(f"Successful conversions: {succeeded}")
    print(f"Failed conversions:     {failed}")
    print(f"Total WAV duration:     {total_duration_s / 60:.1f} minutes")


def _run_tests() -> None:
    result = convert_mp3_to_wav("nonexistent.mp3", "data/processed/audio/wav/test.wav", 16000, 1)
    assert isinstance(result, bool), "convert_mp3_to_wav must return a bool"
    print("Assert passed: return type is bool")

    text = extract_readaloud_text("data/raw/suzBl_readaloud.zip", "071", "MRK", 1)
    assert isinstance(text, str) and len(text) > 0, "expected non-empty string for MRK chapter 1"
    assert any("ऀ" <= ch <= "ॿ" for ch in text), "expected Devanagari characters in result"
    print("Assert passed: extract_readaloud_text returned non-empty Devanagari text")


if __name__ == "__main__":
    main()
