"""
Conversion of the songs of the canaries llb11 and llb16 (Cohen 2022, "Song recordings and annotation files of 3
canaries used to evaluate training of TweetyNet models", https://zenodo.org/records/5956980, CC0; the dataset of llb3)
to the format of canopy.datasets.vocalizations: in canopy/datasets/<bird>/ (not tracked), the annotated recordings
(WAV) and one CSV of annotations per recording, columns "wave", "start", "end", "syll" (one row per syllable, the times
in seconds from the start of the recording). The archive <bird>_data.zip holds <bird>_annot.csv (one row per
syllable: label, onset_s, offset_s, audio_file) and <bird>_songs/*.wav; only the recordings with annotations are kept.

Usage: python -m canopy.datasets.cohen_canaries <path of <bird>_data.zip> <bird>
"""
import io
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import soundfile as sf

from canopy.datasets.songs import DATASETS_DIR, SONGS_SAMPLING_RATE


def convert(archive, bird):
    """Writes the annotated recordings of bird and their annotations in canopy/datasets/<bird>/ from its archive.
    Returns (number of recordings, number of syllables)."""
    target = DATASETS_DIR / bird
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        annotations = pd.read_csv(io.BytesIO(zip_file.read(f"{bird}_annot.csv")))
        annotations["wave"] = annotations["audio_file"].map(lambda path: PurePosixPath(str(path).replace("\\", "/")).name)
        annotations = annotations.rename(columns={"onset_s": "start", "offset_s": "end", "label": "syll"})
        annotations["syll"] = annotations["syll"].astype(str)
        members = {PurePosixPath(name).name: name for name in zip_file.namelist()
                   if name.startswith(f"{bird}_songs/") and name.endswith(".wav")}
        missing = sorted(set(annotations["wave"]) - set(members))
        if missing:
            raise ValueError(f"{len(missing)} annotated recordings not in the archive, e.g. {missing[:3]}")
        for wave, rows in annotations.groupby("wave"):
            path = target / wave  # (the name only: nothing written outside the folder of the bird)
            with zip_file.open(members[wave]) as source, open(path, "wb") as copy:
                copy.write(source.read())
            if sf.info(path).samplerate != SONGS_SAMPLING_RATE:
                raise ValueError(f"{wave}: {sf.info(path).samplerate} Hz, not {SONGS_SAMPLING_RATE}")
            rows.sort_values("start")[["wave", "start", "end", "syll"]].to_csv(path.with_suffix(".csv"), index=False)
    return annotations["wave"].nunique(), len(annotations)


if __name__ == "__main__":
    recordings, syllables = convert(Path(sys.argv[1]), sys.argv[2])
    print(f"{sys.argv[2]}: {recordings} recordings and {syllables} syllables in {DATASETS_DIR / sys.argv[2]}")
