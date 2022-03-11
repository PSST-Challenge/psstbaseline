import csv
import logging
import multiprocessing
import os
from argparse import ArgumentParser
from typing import Iterable, Union

import psstdata
import pyctcdecode
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm

from datahelpers import ASRLogits


def get_args():
    parser = ArgumentParser()
    parser.add_argument("splits", nargs="*", default=("train", "valid", "test"))
    parser.add_argument("--logits-dir", default="./out/logits")
    parser.add_argument("--decode-dir", default="./out/decode")
    parser.add_argument("--beam-width", default=100)
    parser.add_argument("--lm", default=None, help="Filename for a language model in arpa format.")
    parser.add_argument("--n-jobs", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--log-level", default=logging.INFO)
    return parser.parse_args()


def main(
        splits: Iterable[str],
        *,
        logits_dir: str,
        decode_dir: str,
        beam_width: int,
        lm: str,
        n_jobs: int,
        log_level: Union[int, str]
):
    logging.basicConfig(level=log_level)

    decoder = ParallelDecoder(beam_width=beam_width, lm_file=lm, vocabulary=load_vocabulary())

    for split in splits:
        predictions_file = os.path.join(logits_dir, f"logits-{split}.npz")


        if lm and not os.path.exists(lm):
            logging.warning(f"Couldn't find LM file `{lm}`. Skipping.")
            continue
        lm_name = os.path.basename(lm).replace('.arpa', '') if lm else ''
        lm_name = f"-{lm_name}" if lm_name else ""
        logging.info(f"Decoding {split}.")

        logits = ASRLogits.load(predictions_file)
        results = decoder.decode_threaded(logits, n_jobs=n_jobs)

        out_file = os.path.join(decode_dir, f"decoded-{split}{lm_name}.tsv")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            writer = csv.writer(f, dialect=csv.excel_tab)
            writer.writerow(("utterance_id", "asr_transcript"))
            for utterance_id, asr_transcript in sorted(results.items()):
                writer.writerow((utterance_id, asr_transcript))
        logging.info(f"Wrote output to {out_file}")


class ParallelDecoder:
    def __init__(self, lm_file, beam_width, vocabulary):
        self.lm_file = lm_file
        self.beam_width = beam_width
        self.vocabulary = vocabulary

    def __call__(self, args):
        utterance_id, utterance_logits = args
        decoder = build_ctcdecoder(self.vocabulary, self.lm_file)
        asr_transcript = decoder.decode(utterance_logits, beam_width=self.beam_width)
        return utterance_id, asr_transcript

    def decode_threaded(self, logits: ASRLogits, decode_beam_width: int = 100, n_jobs=1, lm_file=None):
        if n_jobs > 1:
            jobs = multiprocessing.Pool(n_jobs).imap_unordered(self, logits.items())
        else:
            jobs = (self(item) for item in logits.items())
        results = {}
        for utterance_id, asr_transcript in tqdm(jobs, total=len(logits)):
            logging.debug(f"{utterance_id}: {asr_transcript}")
            results[utterance_id] = asr_transcript
        return results


def load_vocabulary():
    assert all(idx == n for n, idx in enumerate(psstdata.VOCAB_ARPABET.values()))
    vocabulary = [
        "",  # CTC pad character
        psstdata.UNK,   # The next three indices are used by fairseq but don't seem to come up. Uncertain what
        "?reserved2?",  # they're for. We need to name them for pyctcdecode though.
        "?reserved3?",
    ]
    boundary = pyctcdecode.alphabet.BPE_TOKEN
    vocabulary.extend([
        # Tokens preceded by BPE_TOKEN force a "word" boundary, which decodes how we want.
        boundary + t + boundary
        for t in psstdata.VOCAB_ARPABET
        if t not in (
            psstdata.PAD,
            psstdata.UNK, # It seems fairseq moved this to the beginning, though we don't really use it anyway.
        )
    ])
    for n, symbol in enumerate(vocabulary):
        print(f"|{symbol} |{n:2n} |")
    return vocabulary


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))