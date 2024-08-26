WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
PROCESSOR_NAME = "processor_config.json"
CHAT_TEMPLATE_NAME = "chat_template.json"
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

TRANSFORMER_LAYER_KEY_PREFIX = "model.layers"
QKVO_KEY_TEMPLATE = TRANSFORMER_LAYER_KEY_PREFIX + ".{l}.self_attn.{type}_proj.weight"
MLP_KEY_TEMPLATE = TRANSFORMER_LAYER_KEY_PREFIX + ".{l}.mlp.{type}_proj.weight"
ROTARY_EMB_KEY_TEMPLATE = TRANSFORMER_LAYER_KEY_PREFIX + ".{l}.self_attn.rotary_emb.inv_freq"
LAYERNORM_KEY_TEMPLATE = TRANSFORMER_LAYER_KEY_PREFIX + ".{l}.{type}_layernorm.weight"
BLOCK_TEMPLATE = "{type}.{l}"
ATTN_SAVE_PATH = "l{l}.self_attn.bin"
MLP_SAVE_PATH = "l{l}.mlp.bin"
INPUT_SAVE_PATH = "input.bin"
OUTPUT_SAVE_PATH = "output.bin"
INPUT_EMB_KEY = "model.embed_tokens.weight"
OUTPUT_LAYERNORM_KEY = "model.norm.weight"
OUTPUT_HEAD_KEY = "lm_head.weight"

FILES_TO_SYNC = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
)
