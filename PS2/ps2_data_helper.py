import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import random


class HotpotQAExample(object):
    """ HotpotQA"""
    def __init__(self,
                 qas_id,
                 question_tokens,
                 context_tokens,
                 sentences_label=None,
                 paragraph_label=None):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.context_tokens = context_tokens
        self.sentences_label = sentences_label
        self.paragraph_label = paragraph_label

    def __repr__(self):
        qa_info = "qas_id:{} question:{}".format(self.qas_id, self.question_tokens)
        if self.sentences_label:
            qa_info += " sentence label:{}".format(''.join([str(x) for x in self.sentences_label]))
        if self.paragraph_label:
            qa_info += " paragraph label: {}".format(self.paragraph_label)
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
                 pq_end_pos,
                 cls_label=None,
                 cls_weight=None,
                 is_related=None,
                 roll_back=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_mask = cls_mask
        self.pq_end_pos = pq_end_pos
        self.cls_label = cls_label
        self.cls_weight = cls_weight
        self.is_related = is_related
        self.roll_back = roll_back


def read_second_hotpotqa_examples(args,
                                  input_file,
                                  best_paragraph_file,
                                  related_paragraph_file,
                                  # new_context_file,
                                  is_training: str = 'train',
                                  not_related_sample_rate: float = 0.25):
    """ 获取原始数据 """
    data = json.load(open(input_file, 'r'))
    best_paragraph = json.load(open(best_paragraph_file, 'r'))
    related_paragraph = json.load(open(related_paragraph_file, 'r'))

    examples = []
    related_num = 0
    not_related_num = 0

    for info in tqdm(data, desc="reading examples..."):
        context = info['context']
        question = info['question']
        if is_training == 'test':
            supporting_facts = []
        else:
            supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        qas_id = info['_id']
        best_paragraph_idx = best_paragraph[qas_id]
        best_paragraph_content = ''
        for sent_idx, sent in enumerate(context[best_paragraph_idx][1]):
            best_paragraph_content += sent + ' '
        best_paragraph_content = best_paragraph_content.strip()
        question = question + best_paragraph_content
        for idx, paragraph in enumerate(context):
            if idx == best_paragraph_idx:
                continue
            labels = []
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    labels.append(1)
                    related = True
                else:
                    labels.append(0)
            if is_training == 'train' and not related and random.random() > not_related_sample_rate:
                continue
            if related:
                related_num += 1
            else:
                not_related_num += 1
            example = HotpotQAExample(
                qas_id='{}_{}'.format(qas_id, idx),
                question_tokens=question,
                context_tokens=paragraph,
                sentences_label=labels,
                paragraph_label=related
            )
            examples.append(example)
    print("dataset type: {} related num:{} not related num: {} related / not: {} sample rate: {}".format(
        is_training,
        related_num,
        not_related_num,
        related_num / not_related_num,
        not_related_sample_rate
    ))
    return examples


global_tokenizer = None
global_max_seq_length = None
global_is_training = None
global_cls_token = None
global_sep_token = None
global_unk_token = None
global_pad_token = None


def second_example_process(data):
    """ convert sample to feature """
    example, example_index = data
    features = []
    related_sent_num = 0
    not_related_sent_num = 0
    global global_tokenizer
    global global_max_seq_length
    global global_is_training
    global global_cls_token
    global global_sep_token
    global global_unk_token
    global global_pad_token
    query_tokens = global_tokenizer.tokenize(example.question_tokens)

    if len(query_tokens) >= 400:
        query_tokens = query_tokens[:300]
    # special tokens ['CLS'] ['SEP'] ['SEP']
    max_context_length = global_max_seq_length - len(query_tokens) - 3
    cur_context_length = 0
    query_length = len(query_tokens) + 2
    unique_id = 0
    all_tokens = [global_cls_token] + query_tokens + [global_sep_token]
    query_end_idx = len(all_tokens) - 1
    cls_mask = [1] + [0] * (len(all_tokens) - 1)
    if global_is_training == 'train' or global_is_training == 'dev':
        cls_label = [1 if example.paragraph_label else 0] + [0] * (len(all_tokens) - 1)
    else:
        cls_label = [0] + [0] * (len(all_tokens) - 1)
    cls_weight = [1] + [0] * (len(all_tokens) - 1)
    sent_idx = 0
    pre_sent1_length = None
    pre_sent2_length = None

    while sent_idx < len(example.sentences_label):
        sentence = example.context_tokens[1][sent_idx]
        sent_label = example.sentences_label[sent_idx]
        sentence_tokens = global_tokenizer.tokenize(sentence)
        if sent_label:
            related_sent_num += 1
        else:
            not_related_sent_num += 1
        if len(sentence_tokens) + 1 > max_context_length:
            sentence_tokens = sentence_tokens[:max_context_length - 1]
        roll_back = 0
        if cur_context_length + len(sentence_tokens) + 1 > max_context_length:
            """ add two extra sentences when the length exceeds maximum length """
            context_end_idx = len(all_tokens)
            pq_end_pos = [query_end_idx, context_end_idx]
            all_tokens += [global_sep_token, ]
            tmp_len = len(all_tokens)
            while (len(all_tokens)) < global_max_seq_length:
                all_tokens.append(global_pad_token)
            input_ids = global_tokenizer.convert_tokens_to_ids(all_tokens)
            query_ids = [0] * query_length + [1] * (tmp_len - query_length) + [0] * (global_max_seq_length - tmp_len)
            input_mask = [1] * tmp_len + [0] * (global_max_seq_length - tmp_len)
            cls_mask += [1] + [0] * (global_max_seq_length - tmp_len)
            cls_label += [0] + [0] * (global_max_seq_length - tmp_len)
            cls_weight += [0] + [0] * (global_max_seq_length - tmp_len)
            if pre_sent2_length is not None:
                if pre_sent2_length + pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                    roll_back = 2
                elif pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                    roll_back = 1
            elif pre_sent1_length is not None and pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                roll_back = 1
            sent_idx -= roll_back

            real_related = int(bool(sum(cls_label) - cls_label[0]))
            if real_related != cls_label[0]:
                cls_label[0] = real_related
            assert len(cls_mask) == global_max_seq_length
            assert len(cls_label) == global_max_seq_length
            assert len(cls_weight) == global_max_seq_length
            assert len(input_ids) == global_max_seq_length
            assert len(input_mask) == global_max_seq_length
            assert len(query_ids) == global_max_seq_length
            feature = HotpotInputFeatures(unique_id='{}_{}'.format(example.qas_id, unique_id),
                                          example_index=example_index,
                                          doc_span_index=unique_id,
                                          tokens=all_tokens,
                                          input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=query_ids,
                                          cls_mask=cls_mask,
                                          pq_end_pos=pq_end_pos,
                                          cls_label=cls_label,
                                          cls_weight=cls_weight,
                                          is_related=real_related,
                                          roll_back=roll_back
                                          )
            features.append(feature)
            unique_id += 1

            cur_context_length = 0
            all_tokens = [global_cls_token, ] + query_tokens + [global_sep_token, ]
            query_end_idx = len(all_tokens) - 1
            cls_mask = [1] + [0] * (len(all_tokens) - 1)
            cls_label = [1 if example.paragraph_label else 0] + [0] * (len(all_tokens) - 1)
            cls_weight = [1] + [0] * (len(all_tokens) - 1)
        else:
            all_tokens += [global_unk_token, ] + sentence_tokens  # unk
            cls_mask += [1] + [0] * (len(sentence_tokens) + 0)
            cls_label += [sent_label] + [0] * (len(sentence_tokens) + 0)
            cls_weight += [1 if sent_label else 0.2] + [0] * (len(sentence_tokens) + 0)
            cur_context_length += len(sentence_tokens) + 1
            sent_idx += 1
        pre_sent2_length = pre_sent1_length
        pre_sent1_length = len(sentence_tokens) + 1
    context_end_idx = len(all_tokens)
    pq_end_pos = [query_end_idx, context_end_idx]
    all_tokens += [global_sep_token, ]
    cls_mask += [1]
    cls_label += [0]
    cls_weight += [0]
    tmp_len = len(all_tokens)
    while len(all_tokens) < global_max_seq_length:
        all_tokens.append(global_pad_token)
    input_ids = global_tokenizer.convert_tokens_to_ids(all_tokens)

    query_ids = [0] * query_length + [1] * (tmp_len - query_length) + [0] * (global_max_seq_length - tmp_len)
    input_mask = [1] * tmp_len + [0] * (global_max_seq_length - tmp_len)
    cls_mask += [0] * (global_max_seq_length - tmp_len)
    cls_label += [0] * (global_max_seq_length - tmp_len)
    cls_weight += [0] * (global_max_seq_length - tmp_len)

    real_related = int(bool(sum(cls_label) - cls_label[0]))
    if real_related != cls_label[0]:
        cls_label[0] = real_related
    assert len(cls_mask) == global_max_seq_length
    assert len(cls_label) == global_max_seq_length
    assert len(cls_weight) == global_max_seq_length
    assert len(input_ids) == global_max_seq_length
    assert len(input_mask) == global_max_seq_length
    assert len(query_ids) == global_max_seq_length
    feature = HotpotInputFeatures(unique_id='{}_{}'.format(example.qas_id, unique_id),
                                  example_index=example_index,
                                  doc_span_index=unique_id,
                                  tokens=all_tokens,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=query_ids,
                                  cls_mask=cls_mask,
                                  pq_end_pos=pq_end_pos,
                                  cls_label=cls_label,
                                  cls_weight=cls_weight,
                                  is_related=real_related,
                                  roll_back=0
                                  )
    features.append(feature)
    result = {}
    result["features"] = features
    result["related_sent_num"] = related_sent_num
    result["not_related_sent_num"] = not_related_sent_num
    return result


def convert_examples_to_second_features(examples,
                                        tokenizer,
                                        max_seq_length,
                                        is_training,
                                        cls_token=None,
                                        sep_token=None,
                                        unk_token=None,
                                        pad_token=None
                                        ):
    """ convert_examples_to_second_features """
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
    related_sent_num = 0
    not_related_sent_num = 0
    get_qas_id = {}
    pool_size = max(1, multiprocessing.cpu_count() // 4)
    pool = Pool(pool_size)
    datas = [(example, example_index) for example_index, example in enumerate(examples)]
    for result in tqdm(pool.imap(func=second_example_process, iterable=datas),
                       total=len(datas),
                       desc="Convert examples to features..."):
        features.extend(result["features"])
        related_sent_num += result["related_sent_num"]
        not_related_sent_num += result["not_related_sent_num"]

    print('get feature num:{} related sentences num: {} not related senteces num:{}'.format(len(features),
                                                                                            related_sent_num,
                                                                                            not_related_sent_num))
    return features