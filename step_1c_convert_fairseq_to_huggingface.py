import logging
import os
import re
import tempfile

import torch
from transformers.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2ForCTC


def main(
        fairseq_file="out/psst-18213156/checkpoint_best.pt",
        pretrain_file="pretrained/wav2vec_small.pt",
        out_dir="./rcgale/psst-wav2vec-base"
):
    pretrained = torch.load(pretrain_file)

    fairseq = torch.load(fairseq_file)
    fairseq_model = fairseq.get("model", fairseq)

    huggingface_config_args = {
        k: f(pretrained["args"])
        for k, f in CONFIG_MAPPING.items()
        # if hasattr(fairseq, k)
    }

    huggingface_state = {
        wav2vec2_map_fairseq_to_huggingface(key): value
        for key, value in fairseq_model.items()
    }

    huggingface_config = Wav2Vec2Config(vocab_size=46, **huggingface_config_args)

    with tempfile.TemporaryDirectory() as td:
        temp_file_name = os.path.join(td, "temp.pt")
        torch.save(huggingface_state, temp_file_name)
        huggingface_model = Wav2Vec2ForCTC.from_pretrained(temp_file_name, config=huggingface_config)
        huggingface_model.save_pretrained(out_dir)

    for unused in set(huggingface_state) - set(huggingface_model.state_dict()):
        logging.warning(f"Key {unused} didn't find its way into the huggingface model")

    return huggingface_model, huggingface_config


def wav2vec2_map_fairseq_to_huggingface(key):
    prefix = r"^(?:w2v_encoder\.)?(?:w2v_model\.)?"
    subs = (
        (rf"{prefix}encoder\.layers\.(\d+)\.self_attn\.", r"wav2vec2.encoder.layers.\1.attention."),
        (rf"{prefix}encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"wav2vec2.encoder.layers.\1.layer_norm."),
        (rf"{prefix}encoder\.layers\.(\d+)\.final_layer_norm\.", r"wav2vec2.encoder.layers.\1.final_layer_norm."),
        (rf"{prefix}encoder\.layers\.(\d+)\.fc1\.", r"wav2vec2.encoder.layers.\1.feed_forward.intermediate_dense."),
        (rf"{prefix}encoder\.layers\.(\d+)\.fc2\.", r"wav2vec2.encoder.layers.\1.feed_forward.output_dense."),
        (rf"{prefix}feature_extractor\.conv_layers\.0\.0\.", "wav2vec2.feature_extractor.conv_layers.0.conv."),
        (rf"{prefix}feature_extractor\.conv_layers\.0\.2\.", "wav2vec2.feature_extractor.conv_layers.0.layer_norm."),
        (rf"{prefix}feature_extractor\.conv_layers\.0\.2\.", "wav2vec2.feature_extractor.conv_layers.0.layer_norm."),
        (rf"{prefix}feature_extractor\.conv_layers\.1\.0\.", "wav2vec2.feature_extractor.conv_layers.1.conv."),
        (rf"{prefix}feature_extractor\.conv_layers\.2\.0\.", "wav2vec2.feature_extractor.conv_layers.2.conv."),
        (rf"{prefix}feature_extractor\.conv_layers\.3\.0\.", "wav2vec2.feature_extractor.conv_layers.3.conv."),
        (rf"{prefix}feature_extractor\.conv_layers\.4\.0\.", "wav2vec2.feature_extractor.conv_layers.4.conv."),
        (rf"{prefix}feature_extractor\.conv_layers\.5\.0\.", "wav2vec2.feature_extractor.conv_layers.5.conv."),
        (rf"{prefix}feature_extractor\.conv_layers\.6\.0\.", "wav2vec2.feature_extractor.conv_layers.6.conv."),
        (rf"{prefix}quantizer\.vars\.", r"quantizer.codevectors."),
        (rf"{prefix}mask_emb", "wav2vec2.masked_spec_embed"),
        (rf"{prefix}final_proj\.", "project_hid."),
        (rf"{prefix}post_extract_proj\.", "wav2vec2.feature_projection.projection."),
        (rf"{prefix}encoder\.pos_conv\.0\.", "wav2vec2.encoder.pos_conv_embed.conv."),
        (rf"{prefix}encoder\.layer_norm\.", "wav2vec2.encoder.layer_norm."),
        (rf"{prefix}layer_norm", "wav2vec2.feature_projection.layer_norm"),
        (rf"{prefix}proj\.", "lm_head."),
    )
    out = key
    for find, replace in subs:
        out = re.sub(find, replace, out)
    if out == key:
        logging.warning(f"Unchanged: {out}")
    return out


CONFIG_MAPPING = {
    "transformers_version": lambda c: "4.7.0.dev0",
    "architectures": lambda c: ["Wav2Vec2ForPreTraining"],  # hard-coded for now
    "num_hidden_layers": lambda c: c.encoder_layers,
    "hidden_size": lambda fs: fs.encoder_embed_dim,
    "intermediate_size": lambda fs: fs.encoder_ffn_embed_dim,
    "num_attention_heads": lambda fs: fs.encoder_attention_heads,
    "feat_extract_activation": lambda fs: fs.activation_fn,
    "hidden_dropout": lambda fs: fs.dropout,
    "attention_dropout": lambda fs: fs.attention_dropout,
    "activation_dropout": lambda fs: fs.activation_dropout,
    "layerdrop": lambda fs: fs.encoder_layerdrop,
    "proj_codevector_dim": lambda fs: fs.final_dim,
    "conv_dim": lambda c: tuple(dim for dim, kernel, stride in eval(c.conv_feature_layers)),
    "conv_kernel": lambda c: tuple(kernel for dim, kernel, stride in eval(c.conv_feature_layers)),
    "conv_stride": lambda c: tuple(stride for dim, kernel, stride in eval(c.conv_feature_layers)),
    "num_conv_pos_embeddings": lambda fs: fs.conv_pos,
    "num_conv_pos_embedding_groups": lambda fs: fs.conv_pos_groups,
}


if __name__ == '__main__':
    main()