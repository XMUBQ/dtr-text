import pandas as pd
import os

data_dir='/home/zbohan/dtr-text/data/coauthor-v1.0/'
jss=os.listdir(data_dir)
for jsf in jss:
    df = pd.read_json(data_dir+jsf,lines=True)
    print(df.eventName.value_counts())
    break