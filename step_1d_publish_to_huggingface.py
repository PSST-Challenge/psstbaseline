from transformers.models.wav2vec2 import Wav2Vec2ForCTC


def main(
        model_dir="./out/models-huggingface/psst-wav2vec-base",
        repo_name="rcgale/psst-apr-baseline"
):
    huggingface_model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    huggingface_model.push_to_hub(repo_name)


if __name__ == '__main__':
    main()
