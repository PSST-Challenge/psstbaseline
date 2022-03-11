# psstbaseline

Check back soon!


## Installation

First, `git clone` this repo and `cd` to its directory.

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

We're publishing our model's pretrained weights, and we'd encourage you to do the same. 

```bash
python step_1d_publish_to_huggingface.py
```


## Step 1 (alternate): Download a pre-trained model

...


