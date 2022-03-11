import datetime
import logging
import os
from argparse import ArgumentParser

import psstdata

# This is a max of approximately one minute of audio per batch. Longest sample is slightly shorter.
# Fairseq wav2vec batches by audio duration (samples/sec) rather than number of utterances.
# So a batch could be that one utterance, or it could be 12 5-second utterances.
MAX_TOKENS = 1120000


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-prep-dir",
        default="./out/data/psst-fairseq",
        help="The directory where we'll write the fairseq files"
    )
    parser.add_argument(
        "--manifest-root-dir",
        default=None,
        help="The root dir to put at the top of the manifest file,  if different from the `psstdata` data directory"
    )
    return parser.parse_args()


def main(data_prep_dir: str, manifest_root_dir: str):
    """
    Expected output:
    train: prepared 2298 utterances in ./psst-data-fairseq, totaling 2:46:21.628313 of audio
    valid: prepared 341 utterances in ./psst-data-fairseq, totaling 0:18:30.934000 of audio
    test: prepared 652 utterances in ./psst-data-fairseq, totaling 0:30:30.257812 of audio (once test is released)
    """
    data = psstdata.load()
    manifest_root_dir = manifest_root_dir or os.path.dirname(data.train[0].root_dir)
    os.makedirs(data_prep_dir, exist_ok=True)

    # Dictionary file
    with open(os.path.join(data_prep_dir, "dict.ltr.txt"), "w") as f:
        for arpa, idx in psstdata.VOCAB_ARPABET.items():
            if arpa in (psstdata.PAD, psstdata.UNK):
                continue
            f.write(f"{arpa} {idx}\n")

    # Manifests and transcripts
    for split_name in ("train", "valid", "test"):
        split_data = getattr(data, split_name)
        manifest_filename = os.path.join(data_prep_dir, f"{split_name}.tsv")
        transcript_filename = os.path.join(data_prep_dir, f"{split_name}.ltr")

        with open(manifest_filename, "w") as mf, open(transcript_filename, "w") as tf:
            d = os.path.join(manifest_root_dir, split_name)
            mf.write(f"{d}\n")
            for utterance in split_data:
                if utterance.duration_frames > MAX_TOKENS:
                    logging.warning(f"SKIPPING: {utterance.utterance_id} because duration {utterance.duration_frames} > {MAX_TOKENS}")
                    continue
                mf.write(f"{utterance.filename}\t{utterance.duration_frames}\n")
                tf.write(f"{utterance.transcript}\n")

        audio_seconds = sum(u.duration_frames for u in split_data) / psstdata.WAV_FRAME_RATE
        audio_duration = datetime.timedelta(seconds=audio_seconds)
        print(f"{split_name}: prepared {len(split_data)} utterances in {data_prep_dir}, totaling {audio_duration} of audio")


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))
