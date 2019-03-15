# smile-warrior
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

## Training logs:
Data from 3 training sessions, accordingly for 10, 20 and 30 epochs (without augmentation)
### Accuracy:
![acc](logs1/acc.PNG)

### Loss:
![loss](logs1/loss.PNG)

### Validation accuracy:
![val_acc](logs1/val_acc.PNG)

### Validation loss:
![val_loss](logs1/val_loss.PNG)
