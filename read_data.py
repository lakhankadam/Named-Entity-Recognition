import pandas as pd

def read_data_from_csv():
    data = pd.read_csv('data/ner_dataset.csv', encoding='unicode_escape')
    data.head()
    return data