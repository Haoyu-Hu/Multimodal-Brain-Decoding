import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project_name", type=str)

    parser.add_argument("--run_name", type=str)

    parser.add_argument("--train_path", type=str, default=None, help="Folder with training graph jsons.")

    parser.add_argument("--test_path", type=str, default=None, help="Folder with testing graph jsons.")

    parser.add_argument("--folds", type=int, default=10, help="Default is 10.")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs. Default is 200.")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default is 0.001.")

    parser.add_argument('--lr_decay_factor', type=float, default=0.5)

    parser.add_argument('--lr_decay_step_size', type=int, default=50)

    parser.add_argument("--momentum", type=float, default=0.1)

    parser.add_argument("--weight-decay", type=float, default=5*10**-5, help="Adam weight decay. Default is 5*10^-5.")

    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")

    parser.add_argument("--num_per", type=int, default=16, help="Default is 16")

    parser.add_argument("--epsilon", type=float, default=0.3, help="Default is 0.3.")

    parser.add_argument("--resume", default=False, type=bool)

    parser.add_argument("--beta", default=1e-5, type=float, help="weight of wasserstain loss")

    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')

    # vcm
    parser.add_argument('--topk_ratio', default=0.1, type=float)

    parser.add_argument('--rest_limitation', type=int, default=70000)

    # baseline
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of LSTM layers')

    parser.add_argument('--nheads', type=int, default=1, help='number of heads in transformer multi-head attention')

    parser.add_argument('--modality', type=str, default='multi')

    parser.add_argument('--task', type=str, default='all')

    parser.add_argument('--baseline', type=str, default='lstm')

    # vcm decode
    parser.add_argument("--vcm_path", type=str, default=None)

    parser.add_argument("--data_dict", type=str, default=None)

    parser.add_argument("--index_path", type=str, default=None)

    parser.add_argument("--border_path", type=str, default=None)

    parser.add_argument("--subject", type=str, default='S2')

    # brain->patch
    parser.add_argument("--z_slice", type=int, default=10)

    parser.add_argument("--x_slice", type=int, default=10)

    parser.add_argument("--y_slice", type=int, default=10)

    # main_model parameter
    parser.add_argument("--brain_arch", type=str, default='MLP')

    parser.add_argument("--res_activation", type=str, default='prelu')

    parser.add_argument('--n_layer', type=int, default=6, help='number of post-processing layers')

    parser.add_argument('--n_state', type=int, default=1, help='number of low-level brain states')

    parser.add_argument('--regions', type=int, default=1, help='number of brain regions')

    parser.add_argument('--token_embed_size', type=int, default=512, help='size of a text embedding feature')

    parser.add_argument('--img_embed_size', type=int, default=512, help='size of an image embedding feature')

    # Clip
    parser.add_argument("--clip_arch", type=str, default='ViT-B-32')

    parser.add_argument("--clip_pretrain", type=str, default='laion2b_s34b_b79k')

    # tasnet
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--L', default=20, type=int,
                        help='Length of the filters in samples (40=5ms at 8kHZ)')
    parser.add_argument('--B', default=256, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--H', default=512, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=3, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=8, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=4, type=int,
                        help='Number of repeats')
    parser.add_argument('--C', default=2, type=int,
                        help='Number of speakers')
    parser.add_argument('--norm_type', default='gLN', type=str,
                        choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
    parser.add_argument('--causal', type=int, default=0,
                        help='Causal (1) or noncausal(0) training')
    parser.add_argument('--mask_nonlinear', default='relu', type=str,
                        choices=['relu', 'softmax'], help='non-linear to generate mask')

    return parser.parse_args()
