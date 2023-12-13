import argparse

from cnn import train

LEARNING_RATE: float = 0.01
GAMMA: float = 0.95
BATCH_SIZE: int = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', metavar='parameters')
    args = parser.parse_args()
    parameters_file_name: str = args.parameters

    train(
        lr=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        parameters_file_name=parameters_file_name)


if __name__ == '__main__':
    main()
