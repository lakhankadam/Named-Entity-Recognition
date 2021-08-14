from itertools import chain
from read_data import read_data_from_csv

def get_dictionary_map(data, token_or_tag):
    token2index = {}
    index2token = {}
    if token_or_tag == 'token':
        vocabulary = list(set(data['Word'].to_list()))
    else:
        vocabulary = list(set(data['Tag'].to_list()))
    idx2tok = {idx:tok for idx, tok in enumerate(vocabulary)}
    tok2idx = {tok:idx for idx, tok in enumerate(vocabulary)}
    return tok2idx, idx2tok

def transform_data(data):
    token2idx, idx2token = get_dictionary_map(data, 'token')
    tag2idx, idx2tag = get_dictionary_map(data, 'tag')
    data['Word_idx'] = data['Word'].map(token2idx)
    data['Tag_idx'] = data['Tag'].map(tag2idx)
    data_fillna = data.fillna(method='ffill', axis=0)
    # Groupby and collect columns
    data_group = data_fillna.groupby(
    ['Sentence #'],as_index=False
    )['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
    return data_group
