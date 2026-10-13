"""Command-line options shared by train.py and test.py."""
import os

# Dataset-specific defaults, filled in for any option left unset on the CLI.
_DATASET_DEFAULTS = {
    'levir': {'max_length': 41, 'num_classes': 3},
    'whu':   {'max_length': 26, 'num_classes': 2},
}


def add_data_args(parser):
    # Dataset switch (drives max_length / num_classes defaults and which path
    # options are required)
    parser.add_argument('--dataset', choices=['levir', 'whu'], required=True)

    # Data paths. LEVIR-MCI: --data_folder + --list_path (+ --token_folder).
    #             WHU-CDC:   --data_folder + --caption_json.
    parser.add_argument('--data_folder',  default=None,
                        help='image root containing <split>/{A,B,label}/ (required)')
    parser.add_argument('--list_path',    default=None,
                        help='[levir] folder with train/val/test.txt and vocab.json, '
                             'as written by preprocess_levir_mci.py (required)')
    parser.add_argument('--token_folder', default=None,
                        help='[levir] folder with the per-image token files '
                             '(default: <list_path>/tokens)')
    parser.add_argument('--vocab_file',   default='vocab',
                        help='[levir] vocab file name inside --list_path, without .json')
    parser.add_argument('--caption_json', default=None,
                        help='[whu] caption JSON, e.g. whuCCcaptions.json (required)')
    parser.add_argument('--max_length',   type=int, default=None,
                        help='max caption length incl. <START>/<END> '
                             '(default: 41 for levir, 26 for whu)')
    parser.add_argument('--num_classes',  type=int, default=None,
                        help='number of segmentation classes (default: 3 for levir, 2 for whu)')
    parser.add_argument('--allow_unk',    type=int, default=1)


def add_model_args(parser):
    """Model options. At test time they must match the training configuration."""
    parser.add_argument('--backbone', default='tu-convnext_base',
                        help='segmentation_models_pytorch encoder name')
    parser.add_argument('--d_model',          type=int, default=256)
    parser.add_argument('--n_heads',          type=int, default=8)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--dropout',          type=float, default=0.1)


def apply_dataset_defaults(args, parser=None):
    """Fill max_length / num_classes defaults from --dataset and check that the
    data paths it needs were given."""
    for key, value in _DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    def fail(msg):
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    if args.data_folder is None:
        fail('--data_folder is required (image root containing <split>/{A,B,label}/)')
    if args.dataset == 'levir':
        if args.list_path is None:
            fail('--dataset levir requires --list_path '
                 '(folder with train/val/test.txt and vocab.json from preprocess_levir_mci.py)')
        if args.token_folder is None:
            args.token_folder = os.path.join(args.list_path, 'tokens')
    else:
        if args.caption_json is None:
            fail('--dataset whu requires --caption_json (e.g. whuCCcaptions.json)')
