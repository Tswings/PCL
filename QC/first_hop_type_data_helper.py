import json
import random
from tqdm import tqdm
from sys import getsizeof
import multiprocessing
from multiprocessing import Pool


class HotpotQAExample(object):
    """ HotpotQA"""
    def __init__(self,
                 qas_id,
                 question_tokens,
                 q_type_label=None):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.q_type_label = q_type_label

    def __repr__(self):
        qa_info = "qas_id:{} question:{}".format(self.qas_id, self.question_tokens)
        if self.q_type_label:
            qa_info += " q_type label:{}".format(''.join([str(self.q_type_label)]))
        else:
            qa_info += " q_type label:0"
        return qa_info

    def __str__(self):
        return self.__repr__()


class HotpotInputFeatures(object):
    """ HotpotQA input features to model """
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_mask,
                 cls_label=None,
                 cls_weight=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_mask = cls_mask
        self.cls_label = cls_label
        self.cls_weight = cls_weight



def read_hotpotqa_examples(input_file,
                           is_training: str = 'train',
                           not_related_sample_rate: float = 0.25):
    data = json.load(open(input_file, 'r'))
    #
    # data = data[:100]
    examples = []

    for info in data:
        question = info['question']
        if is_training == 'test':
            label = 0
        else:
            q_type = info['type']
            label = 0 if q_type == 'comparison' else 1

        example = HotpotQAExample(
            qas_id='{}'.format(info['_id']),
            question_tokens=question,
            q_type_label=label,
        )
        examples.append(example)
    return examples


global_tokenizer = None
global_max_seq_length = None
global_is_training = None
global_cls_token = None
global_sep_token = None
global_unk_token = None
global_pad_token = None


def single_example_process(data):
    example, example_index = data
    features = []
    global global_tokenizer
    global global_max_seq_length
    global global_is_training
    global global_cls_token
    global global_sep_token
    global global_unk_token
    global global_pad_token
    query_tokens = global_tokenizer.tokenize(example.question_tokens)
    query_length = len(query_tokens) + 2
    unique_id = 0
    all_tokens = [global_cls_token] + query_tokens + [global_sep_token]
    cls_mask = [1] + [0] * (len(all_tokens) - 1)
    if global_is_training == 'train' or global_is_training == 'dev':
        cls_label = [1 if example.q_type_label else 0] + [0] * (len(all_tokens) - 1)
    else:
        cls_label = [0] * len(all_tokens)
    cls_weight = [1] + [0] * (len(all_tokens) - 1)

    tmp_len = len(all_tokens)
    input_ids = global_tokenizer.convert_tokens_to_ids(all_tokens) + [0] * (global_max_seq_length - tmp_len)
    query_ids = [0] * query_length + [1] * (tmp_len - query_length) + [0] * (global_max_seq_length - tmp_len)
    input_mask = [1] * tmp_len + [0] * (global_max_seq_length - tmp_len)
    cls_mask += [0] * (global_max_seq_length - tmp_len)
    cls_label += [0] * (global_max_seq_length - tmp_len)
    cls_weight += [0] * (global_max_seq_length - tmp_len)

    assert len(cls_mask) == global_max_seq_length
    assert len(cls_label) == global_max_seq_length
    assert len(cls_weight) == global_max_seq_length
    assert len(input_ids) == global_max_seq_length
    assert len(input_mask) == global_max_seq_length
    assert len(query_ids) == global_max_seq_length
    feature = HotpotInputFeatures(unique_id='{}'.format(example.qas_id),
                                  example_index=example_index,
                                  doc_span_index=unique_id,
                                  tokens=all_tokens,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=query_ids,
                                  cls_mask=cls_mask,
                                  cls_label=cls_label,
                                  cls_weight=cls_weight,
                                  )
    features.append(feature)
    result = {}
    result["features"] = features
    del example
    return result


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 is_training,
                                 cls_token=None,
                                 sep_token=None,
                                 unk_token=None,
                                 pad_token=None
                                 ):
    global global_tokenizer
    global global_max_seq_length
    global global_is_training
    global global_cls_token
    global global_sep_token
    global global_unk_token
    global global_pad_token
    global_tokenizer = tokenizer
    global_max_seq_length = max_seq_length
    global_is_training = is_training
    global_cls_token = cls_token
    global_sep_token = sep_token
    global_unk_token = unk_token
    global_pad_token = pad_token
    features = []
    datas = [(example, example_index) for example_index, example in enumerate(examples)]
    for data in tqdm(datas, total=len(datas), desc="Convert examples to features..."):
        result = single_example_process(data)
        features.extend(result["features"])
    print('get feature num:{}'.format(len(features)))
    return features