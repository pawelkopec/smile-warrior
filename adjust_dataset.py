# Script changing labels of emotions to 1 - smile, 0 - no smile and saving changed dataset to new csv file
import pandas as pd
import argparse

HAPPINESS = 3


def main():

    # data_set = pd.read_csv("fer2013.csv")

    parser = argparse.ArgumentParser(description='Input and output csv file')

    parser.add_argument('input', metavar="input csv file in form: filename.csv", type=argparse.FileType('r'),
                        help='Input csv file which will be adjusted')

    parser.add_argument('output', metavar="output csv file in form: filename.csv", type=argparse.FileType('w'),
                        help='Output csv file which will be adjusted')

    args = parser.parse_args()

    data_set = pd.read_csv(args.input)

    print("Adjusting dataset... ")

    # Emotion with label 3 is happiness to we change label to 1 for our smile detector, other emotions are changed to 0.
    for index, row in data_set.iterrows():
        if row['emotion'] == HAPPINESS:
            data_set.at[index, 'emotion'] = 1
        else:
            data_set.at[index, 'emotion'] = 0

    data_set.to_csv(args.output, index=False)

    # data_set.to_csv(smile_warrior_dataset.csv, index=False)

    print("Smile-Warrior dataset created in file %s" % args.output)


if __name__ == "__main__":
    main()
