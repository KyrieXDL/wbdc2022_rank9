DATA_PATH = '../input/data'
BERT_PATH = '../../data/pretrain_model/chinese-macbert-base'

DESC = {
    'tag_id':"int",
    'id': 'byte',
    'category_id': 'int',
    'title': 'byte',
    'asr_text': 'byte',
    'frame_feature': 'bytes'
}

DESC_NOTAG = {
    'id': 'byte',
    'title': 'byte',
    'asr_text': 'byte',
    'frame_feature': 'bytes'
}
