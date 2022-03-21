# PSST Baseline Models

This repo contains the code necessary to reproduce the baseline models for the 
[Post-Stroke Speech Transcription (PSST) Challenge](https://psst.study). We also hope that it can be useful as example
code, or even a starting point for participants.

## Still Preliminary

The baseline models are not yet finalized as of March 11, 2022. All output and results contained within are preliminary.
You can expect finalized results in coming weeks.

## Installation

First, clone this repo and `cd` to its directory.

```bash
git clone https://github.com/PSST-Challenge/psstbaseline.git
cd psstbaseline
```

Some critical dependencies certainly don't work with the latest versions of Python. This model was run on Python 3.8.

```bash
# Do what you like, but if you like pyenv
pyenv virtualenv 3.8.10 psstbaseline \
  && pyenv local psstbaseline

# or if you like conda
conda create --name psstbaseline python=3.8 \
  && conda activate psstbaseline
```

The dependencies are in a requirements file.

```bash
pip install -r requirements.txt
```

## Data

The scripts download the PSST data on-the-fly using [`psstdata`](https://github.com/PSST-Challenge/psstdata), but for 
simplicity's sake, let's make sure it downloads before we embark on the rest of the journey. We can simply open a python
shell session and take care of it right there.

```python
$ python                                                                                                                                           ~/dev/psstbaseline
>>> import psstdata
>>> data = psstdata.load()
# The credentials in /Users/bobby/.config/psstdata/settings.json were missing or incorrect.
# Please enter the PSST username: ********
# Please enter the PSST password: ********
# psstdata INFO: Downloading a new data version: 2022-03-02
# psstdata INFO: Downloaded `train` to /Users/bobby/psst-data/psst-data-2022-03-02.
# psstdata INFO: Downloaded `valid` to /Users/bobby/psst-data/psst-data-2022-03-02.
# psstdata WARNING: 
# psstdata WARNING: The PSST `train` and `valid` sets were downloaded, but `test` is not yet released.
# psstdata WARNING: Once `test` is released, you should only need to re-run this code to retrieve the additional materials.
# psstdata WARNING: Meantime, data labeled `test` is a copy of `valid` as a convenient placeholder.
# psstdata WARNING: 
# psstdata INFO: Loaded data version 2022-03-02 from /Users/bobby/psst-data
```

You'll need the password provided 
[when you sign up for the challenge](https://docs.google.com/forms/d/e/1FAIpQLScwAC3j7NQ2giyFSjrNen6NhmSbnHqdxS915ftZDBRi2SHQtQ/viewform).

Just to make sure, let's take a peek at the first utterance.

```python
>>> first_utterance = data.train[0]
>>> print((first_utterance.utterance_id, first_utterance.transcript, first_utterance.correctness, first_utterance.filename_absolute))
('ACWT02a-BNT01-house', 'HH AW S', True, '/Users/bobby/psst-data/psst-data-2022-03-02/train/audio/bnt/ACWT02a/ACWT02a-BNT01-house.wav')
```

## NOTE: You might want to skip Step 1.

This is the most finnicky and fragile part of the baseline model pipeline. Plus, you don't necessarily have 
to train an ASR model to participate in PSST. If you'd like to skip Step 1, continue on to the 
[instructions for obtaining a pre-trained model.](#step-1-alternate-download-a-pre-trained-model)

## Step 1: ASR model training

### A few remarks on Fairseq

We used Fairseq to train the baseline model. Fairseq's tools are easier than most when you are using their models the 
way Fairseq intended, and for training purposes, we are! (Conveniently, Wav2vec2 was developed in Fairseq.)

However, if you're trying to change model architecture or training approaches, Fairseq is a pretty rigid 
choice. We originally set out to implement the baseline using Huggingface, but it required so many workarounds that it
was extremely difficult to follow what was going on, even as the ones writing it. And we kept finding more. In an effort
to spare participants from wandering the same winding path, we note here a few distinctive features available
in Fairseq that we believe made a substantial difference in model development and performance, especially when working
with just a couple hours of training data:

- Fairseq's batch sampling logic assembles a batch based on audio duration rather than size. So instead of training on
  eight .wav files per batch, we target 8 minutes of audio, which could be over a hundred four-second
  samples, or four one-minute samples.
- Fairseq is set up to freeze all but the output layer at the beginning of training. Wav2vec2 involves a pre-trained
  base model, and randomly initializing a new output layer on top of that is likely to distort the underlying
  embeddings in detrimental ways.

Fortunately, both toolkits have PyTorch available under the hood. And the Huggingface implementation of Wav2vec2 is 
similar enough to Fairseq that we were able to convert the model to Huggingface, which allows us to distribute the 
models using their easy-to-use pretrained model repository.

### 1a) Data Prep

First we write some data files in Fairseq's format. The script for this is 
[step_1a_prepare_psst_for_fairseq.py](step_1a_prepare_psst_for_fairseq.py):

This creates manifest files `(train|test|valid).tsv`, transcript files `(train|test|valid).ltr`, and 
an index-to-phoneme mapping `dict.ltr.txt`.

```bash
python step_1a_prepare_psst_for_fairseq.py
```

### 1b) Fairseq training

Trains the fairseq model.

```bash
python step_1b_train_fairseq_model.py
```

### 1c) Convert to Huggingface model

Maps the layers and arguments and file format for import into Huggingface.

```bash
python step_1c_convert_fairseq_to_huggingface.py
```

### 1d) Publish to Huggingface

We're publishing our model's pretrained weights, and we encourage you to do the same. The huggingface process is
[documented on their web site](https://huggingface.co/welcome).


```bash
# Make sure you have git-lfs installed
# (https://git-lfs.github.com)
git lfs install
git clone https://huggingface.co/username/model_name
```

## Step 1 (alternate): Download a pre-trained model

There's not actually much to do here, because it's all in the code. But here's the pertinent part:

```python
model = Wav2Vec2ForCTC.from_pretrained("rcgale/psst-apr-baseline")
```

That's it!

## Step 2: ASR Logits

In this context, "logits" are a matrix, with the first dimension representing time, and the second dimension 
representing the likelihood of each phoneme for that window of time.

If you run like so, the program will compute logits for the valid and test set, then save them to a numpy-compatible 
file at `out/logits/logits-(valid|test).npz`:

```
python step_2_audio_to_logits.py valid test
```

You can specify any of `train`, `valid`, and `test`, and if you leave off the arguments, it'll compute logits for all
three.

The training algorithm uses a CTC loss, which  implies [special rules for decoding](https://distill.pub/2017/ctc/). For 
this model, Fairseq reserves the 0th phoneme for the CTC `<pad>` character, then the next three for special purposes 
like unknown and separator annotations, which have less relevance for us, since we are paying no regard to word 
boundaries. Otherwise, the symbols and their indices laid out in the table below.

| Phoneme | Index | Phoneme | Index| Phoneme     | Index|
|--------|-------|---|------|-------------|------|
| AA   | 4 | F |18 | R           |32 |
| AE   | 5 | G |19 | S           |33 |
| AH   | 6 | HH |20 | SH          |34 |
| AO   | 7 | IH |21 | T           |35 |
| AW   | 8 | IY |22 | TH          |36 |
| AY   | 9 | JH |23 | UH          |37 |
| B    |10 | K |24 | UW          |38 |
| CH   |11 | L |25 | V           |39 |
| D    |12 | M |26 | W           |40 |
| DH   |13 | N |27 | Y           |41 |
| DX   |14 | NG |28 | Z           |42 |
| EH   |15 | OW |29 | ZH          |43 |
| ER   |16 | OY |30 | &lt;sil&gt; |44 |
| EY   |17 | P |31 | &lt;spn&gt; |45 |





























## Step 3: ASR Decoding

Decoding is the process of determining the most likely sequence represented by the logits. The command's syntax is 
similar to Step 2, with the split names optional.

```
python step_2_audio_to_logits.py valid test
```

This writes tab-separated files to `out/decode/decoded-(train|valid|test).tsv` containing the best estimation of what's
being said in each utterance. Here's a sample of the first four lines of output from the train split:

|utterance_id             | asr_transcript          |
|-------------------------|-------------------------|
|ACWT02a-BNT01-house      | HH AW S                 |
|ACWT02a-BNT02-comb       | K OW M                  |
|ACWT02a-BNT03-toothbrush | T UW TH B R AH SH       |
|ACWT02a-BNT04-octopus    | AA S AH P R OW G P UH S |

How nice, these four are all correct!

## Step 4: ASR Evaluation

Coming soon. While the models and results are still preliminary, so far we've seen the phoneme error rate (PER) for the 
validation split is usually about 23-24%.

## Step 5: Correctness

Coming soon


