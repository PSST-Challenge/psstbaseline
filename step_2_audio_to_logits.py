import logging
import multiprocessing
import os
import sys
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
from typing import Union

import psstdata
import soundfile
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForCTC

from datahelpers import ASRLogits


def get_args():
    parser = ArgumentParser()
    parser.add_argument("splits", nargs="*", default=("train", "valid", "test"))
    parser.add_argument("--model-name", default="./out/models-huggingface/psst-wav2vec-base")
    parser.add_argument("--logits-dir", default="./out/logits")
    parser.add_argument("--n-jobs", default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--log-level", default=logging.INFO)
    return parser.parse_args()


def main(
        splits,
        model_name: str,
        logits_dir: str,
        n_jobs: int,
        log_level: Union[int, str]
):
    logging.basicConfig(level=log_level)

    n_jobs = n_jobs or multiprocessing.cpu_count() - 1

    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=psstdata.WAV_FRAME_RATE)
    predictor = Predictor(model, feature_extractor)

    data = psstdata.load()

    for split in splits:
        logging.info(f"Predicting {split} split.")
        data_split = getattr(data, split)
        split_logits = predictor.predict_threaded(data_split, n_jobs=n_jobs)
        out_file = os.path.join(logits_dir, f"logits-{split}.npz")
        split_logits.save(out_file)
        logging.info(f"Wrote file to {out_file}")


class Predictor:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor

    def __call__(self, utterance):
        with torch.no_grad():
            raw_speech, fs = soundfile.read(utterance.filename_absolute)
            input = self.feature_extractor(raw_speech, return_tensors="pt", sampling_rate=psstdata.WAV_FRAME_RATE)
            output = self.model(input["input_values"])
            return utterance.utterance_id, output["logits"][0].numpy()

    def predict_threaded(self, data, n_jobs):
        jobs = ThreadPool(n_jobs).imap_unordered(self, data)
        logits = dict(sorted(tqdm(jobs, total=len(data), file=sys.stdout)))
        return ASRLogits(logits)


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))