# Smile-warrior
This repo contains code used for smile detector for project of Gradient Science Club.

# To launch web app:
```
py app.py --cascade  haarcascade_frontalface_default.xml --model  model.hdf5 --weight  weights.03.hdf5
```

## To create dataset:
Run:
```
   python adjust_dataset.py 
```

Result will be a csv file with dataset for Smile-Warrior (by default in: "smile_warrior_dataset.csv").

## To load dataset:
Use:
```
   python data_server.py
```
To split whole dataset for Smile-Warrior into respectively: x_train, y_train, x_test, y_test, x_validate and y_validate datasets. 
