# smile-warrior
This repo contains code used for smile detector for project of Gradient Science Club.

To create dataset for smile-warrior you need to run 'Adjust_dataset' script, which creates it from fer2013.csv data set. As a result you will receive csv file named 'smile_warrior_dataset'.

After it you can load smile-warrior dataset to your project with function Load_dataset which returns respectively: X_train, Y_train, X_test, Y_test, X_validate and Y_validate datasets. 