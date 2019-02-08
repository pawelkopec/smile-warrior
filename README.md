# smile-warrior
This repo contains code used for smile detector for project of Gradient Science Club.

##To create dataset run: 
```bash
    adjust_dataset.py 
```

Result will be csv file named 'smile_warrior_dataset'.

To load dataset use function:\
    load_dataset  
from module:\
    data_server.py
    
Which returns respectively: X_train, Y_train, X_test, Y_test, X_validate and Y_validate datasets. 
