# coding=utf-8
# Adapted based on the HuggingFace Transformers directory: https://github.com/huggingface/transformers/.
#

""" T6 model configuration """


import logging

from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

TPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tpt-small": "configs/tpt-small-config.json",
    "tpt-small-discrete": "configs/tpt-small-discrete-config.json",
}


class TPTConfig(PretrainedConfig):
    r"""
        :class:`~transformers.TPTConfig` is the configuration class to store the configuration of a
        `TPTModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TPTModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `TPTModel`.
            initializer_factor: A factor for initializing all weight matrices (should be kept to 1.0, used for initialization testing).
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = TPT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "tpt"

    def __init__(
        self,
        vocab_size=32128,
        n_positions=512,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        use_tp_enc=False,
        use_tp_dec=False,
        use_discrete_roles=False,
        roles_after_residual=False,
        multi_head_rar=True,
        tpr_binding_type='hadamard',
        role_weights_input='input',
        num_roles=20,
        share_roles_among_heads=True,
        use_tpr_gate=False,
        tpr_output_type='add',
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor

        self.use_tp_enc = use_tp_enc
        self.use_tp_dec = use_tp_dec
        self.use_discrete_roles = use_discrete_roles
        self.num_roles = num_roles
        self.roles_after_residual = roles_after_residual
        self.multi_head_rar = multi_head_rar
        self.tpr_binding_type = tpr_binding_type
        self.role_weights_input = role_weights_input
        self.share_roles_among_heads = share_roles_among_heads
        self.use_tpr_gate = use_tpr_gate
        self.tpr_output_type = tpr_output_type

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers
