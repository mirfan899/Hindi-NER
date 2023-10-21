### Hindi-NER
This is an example project to let you use your dataset and publish it on hugginface.

The dataset used in this project is IJNLP 2008 Hindi dataset. I have converted to `word tag` format, separated by 
tab.

Now split dataset into train, test and validation. You can choose whatever percentange you want for your dataset.

```shell
python split_hindi.py
```

Now you can publish your dataset by adding token and your account info in `publish_hindi.py` script.