import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None, help="Folder with training graph jsons.")

    parser.add_argument("--ng_data", type=str, default=None, help="Folder with training graph jsons.")

    parser.add_argument("--folds", type=int, default=10, help="Default is 10.")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs. Default is 200.")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default is 0.001.")

    parser.add_argument('--lr_decay_factor', type=float, default=0.5)

    parser.add_argument('--lr_decay_step_size', type=int, default=50)

    parser.add_argument("--weight-decay", type=float, default=5*10**-5, help="Adam weight decay. Default is 5*10^-5.")

    parser.add_argument("--batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--num_per", type=int, default=16, help="Default is 16")

    parser.add_argument("--epsilon", type=float, default=0.3, help="Default is 0.3.")

    parser.add_argument("--resume", default=False, type=bool)

    parser.add_argument("--beta", default=1e-5, type=float, help="weight of wasserstain loss")

    parser.add_argument("--classes", type=int, default=1, help="num of classification classes")

    parser.add_argument('--semantic', type=int, default=1, help='dimension of semantic data')

    parser.add_argument('--n_state', type=int, default=1, help='number of low-level brain states')

    parser.add_argument('--regions', type=int, default=1, help='number of brain regions')

    parser.add_argument('--time_points', type=int, default=1, help='number of time points')

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

    return parser.parse_args()
