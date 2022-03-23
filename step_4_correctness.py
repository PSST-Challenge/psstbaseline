import csv
import logging
import os
from argparse import ArgumentParser

import psstdata

import pssteval


def get_args():
    parser = ArgumentParser()
    parser.add_argument("splits", nargs="*", default=("train", "valid", "test"))
    parser.add_argument("--decode-dir", default="./out/decode")
    parser.add_argument("--correctness-dir", default="./out/correctness")
    parser.add_argument("--log-level", default=logging.INFO)
    return parser.parse_args()


def main(splits, decode_dir, correctness_dir, log_level):
    logging.basicConfig(level=log_level)
    data = psstdata.load()
    for split in splits:
        logging.info(f"Running split `{split}`")
        split_data = getattr(data, split)

        decoded_filename = os.path.join(decode_dir, f"decoded-{split}.tsv")
        asr_output = pssteval.load_asr_output(decoded_filename)

        out_file = os.path.join(correctness_dir, f"correctness-{split}.tsv")
        os.makedirs(correctness_dir, exist_ok=True)

        with open(out_file, "w") as f:
            writer = csv.writer(f, dialect=csv.excel_tab)

            # Only two columns (`utterance_id`, `prediction`) are required, but more is nice for analysis.
            writer.writerow(("utterance_id", "truth", "prediction", "transcript", "asr_transcript"))

            for row in asr_output:
                utterance = split_data[row.utterance_id]
                found = target_in_transcript(utterance.prompt, row.asr_transcript)
                writer.writerow(
                    (row.utterance_id, utterance.correctness, found, utterance.transcript, row.asr_transcript)
                )

        logging.info(f"Wrote output to {out_file}")


def target_in_transcript(prompt: str, transcript: str):
    targets = psstdata.ACCEPTED_PRONUNCIATIONS[prompt]
    asr_tokens = transcript.split()
    for target in sorted(targets, key=lambda t: -1 * len(t)):
        target_tokens = target.split()
        for i in range(1 + len(asr_tokens) - len(target_tokens)):
            if target_tokens == asr_tokens[i:i+len(target_tokens)]:
                logging.debug(f"(True)  Found target /{target}/ in transcript /{transcript}/")
                return True
    logging.debug(f"(False) Couldn't find an acceptable form of /{prompt}/ in transcript /{transcript}/")
    return False


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))