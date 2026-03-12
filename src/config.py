class ModelConfig:
    """Model configuration matching DeepFilterNet3 defaults."""

    def __init__(self):
        # Audio parameters
        self.sr = 48000  # Sample rate
        self.fft_size = 960  # FFT size in samples
        self.hop_size = 480  # STFT hop size in samples
        self.nb_erb = 32  # Number of ERB bands
        self.nb_df = 96  # Number of DF bins (DF applied from 0 to nb_df-1)

        # Normalization
        self.norm_tau = 1.0  # Normalization decay factor

        # LSNR (Local SNR) parameters
        self.lsnr_max = 35  # LSNR maximum value (ground truth truncation)
        self.lsnr_min = -15  # LSNR minimum value (ground truth truncation)

        # ERB parameters
        self.min_nb_freqs = 2  # Minimum number of frequency bins per ERB band

        # DF parameters
        self.df_order = 5  # Deep filtering order
        self.df_lookahead = (
            2  # Deep filtering look-ahead (from config: df_lookahead = 2)
        )
        self.pad_mode = "output"  # Padding mode: 'input' or 'output'

        # Convolution parameters
        self.conv_lookahead = (
            2  # Convolution lookahead (from config: conv_lookahead = 2)
        )
        self.conv_ch = 64  # Number of convolution channels (from config: conv_ch = 64)
        self.conv_depthwise = True  # Use depthwise convolutions
        self.convt_depthwise = (
            False  # Use depthwise transposed convolutions (from config)
        )
        self.conv_kernel = (1, 3)  # Convolution kernel size
        self.convt_kernel = (1, 3)  # Transposed convolution kernel size
        self.conv_kernel_inp = (3, 3)  # Input convolution kernel size

        # Embedding/Encoder GRU parameters
        self.emb_hidden_dim = 256  # Embedding hidden dimension
        self.emb_num_layers = (
            3  # Number of embedding layers (from config: emb_num_layers = 3)
        )
        self.emb_gru_skip = "none"  # Embedding GRU skip connection type
        self.emb_gru_skip_enc = "none"  # Encoder GRU skip connection type

        # Deep Filtering decoder GRU parameters
        self.df_hidden_dim = 256  # DF hidden dimension
        self.df_num_layers = 2  # Number of DF layers (from config: df_num_layers = 2)
        self.df_gru_skip = (
            "groupedlinear"  # DF GRU skip (from config: df_gru_skip = groupedlinear)
        )
        self.df_pathway_kernel_size_t = (
            5  # DF pathway kernel size (from config: df_pathway_kernel_size_t = 5)
        )

        # Linear layer parameters
        self.lin_groups = 16  # Linear groups (from config: linear_groups = 16)
        self.enc_lin_groups = (
            32  # Encoder linear groups (from config: enc_linear_groups = 32)
        )

        # Other architecture parameters
        self.enc_concat = False  # Whether to concatenate encoder features
        self.df_n_iter = 1  # Number of DF iterations

        # Post-processing
        self.mask_pf = False  # Post-filter mask
        self.pf_beta = 0.02  # Post-filter beta parameter
        self.lsnr_dropout = False  # LSNR dropout during training

        # Li-GRU specific (not in original config)
        self.batch_size = 1  # Batch size for Li-GRU initialization


# Create default config instance
config = ModelConfig()
