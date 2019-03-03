import pandas as pd
import argparse

HAPPINESS = 3


def main():
    """
    Run this script to create new csv file with dataset for Smile-Warrior from fer2013.csv dataset for Facial
    Expression Recognition.

    This script is changing labels of emotions in fer2013.csv to 1 - smile, 0 - no smile.

    To run correctly you need to pass two paths to csv files as a command-line arguments, first for input file which
    will be adjusted for Smile-Warrior needs and second for output file where new dataset will be created.

    """

    # Parser with paths to csv files from command-line arguments
    args = parse_args()
    data_set = pd.read_csv(args.input)

    print("Adjusting dataset... ")

    # Emotion with label 3 is happiness so we change label to 1 for our smile detector, other emotions are
    # changed to 0.
    for index, row in data_set.iterrows():
        if row['emotion'] == HAPPINESS:
            data_set.at[index, 'emotion'] = 1
        else:
            data_set.at[index, 'emotion'] = 0

    data_set.to_csv(args.output, index=False)
    print("Smile-Warrior dataset created in file %s" % args.output)


def parse_args():
    """
    Function to get paths to csv files from command-line arguments
    """

    parser = argparse.ArgumentParser(description='Input and output csv files')

    parser.add_argument('--input', metavar="input csv file", type=argparse.FileType('r'),
                        help='Path to input csv file which will be adjusted for Smile-Warrior needs.', default="fer2013.csv")

    parser.add_argument('--output', metavar="output csv file", type=argparse.FileType('w'),
                        help='Path to output csv file where new dataset will be created.',
                        default='smile_warrior_dataset.csv')

    return parser.parse_args()


if __name__ == "__main__":
    main()
