# Script changing labels of emotions to 1 - smile, 0 - no smile and saving changed dataset to new csv file
import pandas as pd

HAPPINESS = 3


def main():

    data_set = pd.read_csv("fer2013.csv")

    print("Adjusting dataset... ")

    # Emotion with label 3 is happines to we change label to 1 for our smile detector, other emotions are changed to 0.
    for index, row in data_set.iterrows():
        if row['emotion'] == HAPPINESS:
            data_set.at[index, 'emotion'] = 1
        else:
            data_set.at[index, 'emotion'] = 0

    data_set.to_csv('smile_warrior_dataset.csv', index=False)
    print("Smile-Warrior dataset created in file 'smile_warrior_dataset.csv'")


if __name__ == "__main__":
    main()
