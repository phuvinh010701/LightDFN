"""Configuration for LightDeepFilterNet model."""


class ModelConfig:
    def __init__(self):
        # Audio parameters
        self.sr = 48000
        self.fft_size = 960
        self.hop_size = 480
        self.nb_erb = 32
        self.nb_df = 96

        # Encoder/Decoder parameters
        self.conv_ch = 16
        self.conv_kernel = (1, 3)
        self.convt_kernel = (1, 3)
        self.conv_kernel_inp = (3, 3)

        # GRU parameters
        self.emb_hidden_dim = 256
        self.emb_num_layers = 2
        self.df_hidden_dim = 256
        self.df_num_layers = 3

        # DF parameters
        self.df_order = 5
        self.df_lookahead = 0
        self.df_pathway_kernel_size_t = 1

        # Linear layer parameters
        self.lin_groups = 1
        self.enc_lin_groups = 16

        # Skip connections
        self.emb_gru_skip = "none"
        self.emb_gru_skip_enc = "none"
        self.df_gru_skip = "none"

        # Other
        self.enc_concat = False
        self.mask_pf = False
        self.pf_beta = 0.02
        self.lsnr_min = -10
        self.lsnr_max = 35
        self.lsnr_dropout = False

        # Batch size for Li-GRU
        self.batch_size = 1
