import argparse
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description='Download Places365 dataset')
    parser.add_argument('root', metavar='DIR', help='path to where dataset should be downloaded')
    parser.add_argument('--small', action='store_true', help='Use small version of Places')
    args = parser.parse_args()
    _ = datasets.places365.Places365(root=args.root, split='train-standard',
                                     small=args.small, download=True)


if __name__ == '__main__':
    main()
