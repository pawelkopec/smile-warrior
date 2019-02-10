import pandas as pd
import argparse

HAPPINESS = 3

# Alternative for users who don't pass arguments from command-line, that alternative will be deleted in future version
path_input_file = "fer2013.csv"
path_output_file = 'smile_warrior_dataset.csv'

# Set that flag to 1 if you want to pass paths for csv files as a command-line arguments
# Set that flag to 0 if you want to change paths for csv files by changing global variables above
path_from_terminal = 1


def main():
    """
    Run this script to create new csv file with dataset for Smile-Warrior from fer2013.csv dataset for Facial
    Expression Recognition.

    This script is changing labels of emotions in fer2013.csv to 1 - smile, 0 - no smile.

    To run correctly you need to pass two paths to csv files as a command-line arguments, first for input file which
    will be adjusted for Smile-Warrior needs and second for output file where new dataset will be created.

    Example:

        $ python adjust_dataset.py fer2013.csv smile_warrior_dataset.csv

    """

    if path_from_terminal == 0:
        # Path to csv file defined by user in code
        data_set = pd.read_csv(path_input_file)

    elif path_from_terminal == 1:
        # Path to csv file as command-line arguments
        args = parse_args()
        data_set = pd.read_csv(args.input)

    print("Adjusting dataset... ")

    # Emotion with label HAPPINESS is happiness so we change label to 1 for our smile detector, other emotions are
    # changed to 0.
    for index, row in data_set.iterrows():
        if row['emotion'] == HAPPINESS:
            data_set.at[index, 'emotion'] = 1
        else:
            data_set.at[index, 'emotion'] = 0

    if path_from_terminal == 0:
        data_set.to_csv(path_output_file, index=False)
        print("Smile-Warrior dataset created in file %s" % path_output_file)

    elif path_from_terminal == 1:
        data_set.to_csv(args.output, index=False)
        print("Smile-Warrior dataset created in file %s" % args.output)


def parse_args():
    """
    Function to get paths to csv files from command-line arguments

    Example:

        args = parse_args()

    :return:
        Parser with two arguments 'input' and 'output'.

        First argument 'input' is path to csv file with dataset which will be adjusted for Smile-Warrior needs.
        Second argument 'output' is path to csv file where new dataset will be created.
    """

    parser = argparse.ArgumentParser(description='Input and output csv file')

    parser.add_argument('input', metavar="input csv file in form: filename.csv", type=argparse.FileType('r'),
                        help='Input csv file which will be adjusted')

    parser.add_argument('output', metavar="output csv file in form: filename.csv", type=argparse.FileType('w'),
                        help='Output csv file which will be adjusted')

    return parser.parse_args()


if __name__ == "__main__":
    main()
