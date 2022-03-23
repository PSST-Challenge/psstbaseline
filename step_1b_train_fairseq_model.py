import math
import os
import urllib.request
from argparse import ArgumentParser
from datetime import datetime

import fairseq.options
import fairseq_cli.train
import torch

# This is a max of approximately one minute of audio per batch. Longest sample is slightly shorter.
# Fairseq wav2vec batches by audio duration (samples/sec) rather than number of utterances.
# So a batch could be that one utterance, or it could be 12 5-second utterances.
MAX_TOKENS = 1120000


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--weight-decay", default=0.0)
    return parser.parse_args()


def fine_tune(
        prepared_data="./out/data/psst-fairseq",
        models_dir=f"./out/models",
        save_dir_symlink=f"./out/models/psst-baseline",
        pretrained_model="wav2vec_small.pt",
):
    """
    Still a preliminary model, since OOM issues were preventing the whole training set from being used.
    """
    my_args = get_args()

    parser = fairseq.options.get_training_parser()

    save_dir = get_save_dir(models_dir)
    pretrained_path = download_base_model(pretrained_model)

    in_args = input_args(**{
        "save_dir": save_dir,
        "tensorboard_logdir": f"tensorboard/{os.path.basename(save_dir)}",
        "w2v_path": pretrained_path,
        "data": prepared_data,
        **vars(my_args)
    })
    fairseq_args = fairseq.options.parse_args_and_arch(parser, input_args=in_args)
    fairseq_cli.train.main(fairseq_args)
    if os.path.islink(save_dir_symlink):
        os.unlink(save_dir_symlink)
    os.symlink(os.path.relpath(save_dir, os.path.dirname(save_dir_symlink)), save_dir_symlink)


def get_save_dir(models_dir):
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", None))
    if job_id is None:
        job_id = datetime.now().strftime("%m_%d_%Y_%H_%M")
    save_dir = os.path.join(models_dir, f"psst-{job_id}")
    os.makedirs(save_dir, exist_ok=False)
    return save_dir


def download_base_model(filename, pretrained_dir="./pretrained"):
    pretrained_path = os.path.join(pretrained_dir, filename)
    if not os.path.exists(pretrained_path):
        print(f"Downloading {filename} to {pretrained_dir}")
        os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
        urllib.request.urlretrieve(
            f"https://dl.fbaipublicfiles.com/fairseq/wav2vec/{filename}",
            pretrained_path
        )
    return pretrained_path


def input_args(
        save_dir,
        tensorboard_logdir,
        data,
        w2v_path,
        empty_cache_freq=0,
        best_checkpoint_metric="uer",  # UER is a character error rate, oddly named
        n_gpu=None,
        lr=5e-05,
        max_update=12000,
        warmup_updates=4000,
        hold_updates=4000,
        decay_updates=4000,
        freeze_updates=2000,  # Freeze all but the output layer
        validate_after_updates=1000,
        validate_interval=1,
        valid_subset="valid",
        max_tokens=MAX_TOKENS,   # per device. 1250000 fills up the rtx2080 11GB
        max_tokens_per_minibatch=6400000,
        mask_prob=0.65,
        mask_channel_length=64,
        mask_channel_prob=0.25,
        update_freq=None,
        layerdrop=0.1,  # Unconfirmed but I think it's layerdrop/hidden_dropout for fairseq -> huggingface
        final_lr_scale=0.05,
        final_dropout=0.1,
        dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        random_seed=5728395,
        adam_epsilon=1e-08,
        adam_betas=(0.9, 0.98),
        weight_decay=0.0
):
    # This is pretty clunky, but still less clunky than bash imo
    n_gpu = n_gpu or torch.cuda.device_count() or 1
    update_freq = update_freq or math.ceil(max_tokens_per_minibatch / max_tokens / n_gpu)
    args_string = f"""
      {data}
      --fp16
      --save-dir {save_dir}
      --tensorboard-logdir {tensorboard_logdir}
      --distributed-world-size {n_gpu}
      --empty-cache-freq {empty_cache_freq}
      --max-sample-size {max_tokens}
      --max-tokens {max_tokens}
      --max-tokens-valid {max_tokens}
      --update-freq [{update_freq}]
      --post-process letter
      --valid-subset {valid_subset}
      --no-epoch-checkpoints
      --best-checkpoint-metric {best_checkpoint_metric}
      --num-workers 1
      --max-update {max_update}
      --sentence-avg
      --task audio_pretraining
      --arch wav2vec_ctc
      --w2v-path {w2v_path}
      --labels ltr
      --weight-decay {weight_decay}
      --apply-mask
      --mask-selection static
      --mask-other 0
      --mask-length 10
      --mask-prob {mask_prob}
      --mask-channel-selection static
      --mask-channel-other 0
      --mask-channel-length {mask_channel_length}
      --mask-channel-prob {mask_channel_prob}
      --zero-infinity
      --feature-grad-mult 0.0
      --freeze-finetune-updates {freeze_updates}
      --validate-after-updates {validate_after_updates}
      --validate-interval {validate_interval}
      --optimizer adam
      --adam-betas {adam_betas}
      --adam-eps {adam_epsilon}
      --lr {lr}
      --lr-scheduler tri_stage
      --warmup-steps {warmup_updates}
      --hold-steps {hold_updates}
      --decay-steps {decay_updates}
      --final-lr-scale {final_lr_scale}
      --final-dropout {final_dropout}
      --dropout {dropout}
      --layerdrop {layerdrop}
      --activation-dropout {activation_dropout}
      --criterion ctc
      --attention-dropout {attention_dropout}
      --seed {random_seed}
      --log-format json
      --log-interval 50
      --ddp-backend no_c10d
    """

    args_string_print = args_string.replace("\n", " \\\n").strip().strip("\\")
    print(f"fairseq command:\n  python train.py {args_string_print}\n")

    args = [
    ]
    for line in args_string.split("\n"):
        if not line.strip():
            continue
        flag, *value = line.split(maxsplit=1)
        args.append(flag.strip())
        if len(value) == 1:
            args.append(value[0].strip())
    return args


if __name__ == '__main__':
    fine_tune()

