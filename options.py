import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='arguments for getting the graphs',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mot_seq_size', type=int, default=15, help='size of each motion sequence')
    parser.add_argument('--start_node', type=int, default=0, help='start node for motion sequence')
    parser.add_argument('--div', type=int, default=5, help='determines the sample taken from a motion sequence')
    parser.add_argument('--label', type=int, default=0, help='label for the motion sequence')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for edge weight')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for edge weight')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for edge weight')
    return parser.parse_args()