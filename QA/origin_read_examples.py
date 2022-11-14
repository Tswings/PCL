import json
from tqdm import tqdm
from multiprocessing.pool import Pool
import multiprocessing
from origin_reader_helper import HotpotQAExample, is_whitespace


def question_text_process(question, tokenizer):
    question_tokens = []
    doc_tokens = []
    prev_is_whitespace = True
    for idx, q_ch in enumerate(question):
        if is_whitespace(q_ch):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(q_ch)
            else:
                doc_tokens[-1] += q_ch
            prev_is_whitespace = False
    for d_token in doc_tokens:
        sub_tokens = tokenizer.tokenize(d_token)
        for ind_st, sub_token in enumerate(sub_tokens):
            question_tokens.append(sub_token)
    return question_tokens

is_training_global = False
tokenizer_global = None
sp_dict_global = None


def process_single_data(data):
    global is_training_global
    global tokenizer_global
    global sp_dict_global
    tmp_examples = []
    context = ""
    qas_id = data['_id']
    question = data['question']
    answer = data.get('answer', '')
    sup = data.get('supporting_facts', [])
    length = len(context)
    sent_cls = []
    start_position = None
    end_position = None
    full_sents_mask = []
    full_sents_lbs = []
    char_to_matrix = []
    input_sentence_idx = 1
    question_tokens = question_text_process(question, tokenizer=tokenizer_global)
    if is_training_global:
        answer_label = data['labels'][0]
        sent_lbs = []
        if answer.lower() == 'yes':
            start_position = -1
            end_position = -1
        if answer.lower() == 'no':
            start_position = -2
            end_position = -2
    for ind_con, con in enumerate(data['context']):
        if ind_con in sp_dict_global[qas_id]:
            full_sents_mask += [1 for con1 in con[1] if con1.strip() != '']
        else:
            full_sents_mask += [0] * len(con[1])
        for indc1, c1 in enumerate(con[1]):
            if ind_con in sp_dict_global[qas_id] and c1.strip() == '':
                continue
            if [con[0], indc1] in sup:
                full_sents_lbs.append(1)
            else:
                full_sents_lbs.append(0)
        if is_training_global and con[0] == answer_label[0] and ind_con not in sp_dict_global[qas_id]:
            break
        if ind_con in sp_dict_global[qas_id]:
            offset = 0
            for inds, sent in enumerate(con[1]):
                if sent.strip() == '':
                    continue
                if context == '':
                    white1 = 0
                    while is_whitespace(sent[0]):
                        white1 += 1
                        sent = sent[1:]
                    offset += white1
                elif not sent.startswith(' '):
                    context += ' '
                    length += 1
                    char_to_matrix += [input_sentence_idx]
                context += sent
                char_to_matrix += [input_sentence_idx] * len(sent)
                input_sentence_idx += 1
                sent_cls.append(length)
                if is_training_global and con[0] == answer_label[0]:
                    if offset <= answer_label[1] < offset + len(sent):
                        start_position = length + answer_label[1] - offset  # +5
                    if offset < answer_label[2] <= offset + len(sent):
                        end_position = length + answer_label[2] - offset  # +5
                length += len(sent)
                offset += len(sent)
                if is_training_global:
                    if [con[0], inds] in sup:
                        sent_lbs.append(1)
                    else:
                        sent_lbs.append(0)
    if is_training_global and start_position is None:
        return tmp_examples
    doc_tokens = []
    char_to_newchar = []
    prev_is_whitespace = True
    char_count = -2
    for ch in context:
        if is_whitespace(ch):
            prev_is_whitespace = True
            char_to_newchar.append(char_count + 1)
        else:
            if prev_is_whitespace:
                doc_tokens.append(ch)
                char_count += 2
            else:
                doc_tokens[-1] += ch
                char_count += 1
            prev_is_whitespace = False
            char_to_newchar.append(char_count)
    if prev_is_whitespace:
        char_to_newchar.append(char_count + 1)
    context = ' '.join(doc_tokens)
    sent_cls_n = []
    newchar_to_matrix = [0] * len(context)
    for ind_ctm, ctm in enumerate(char_to_matrix):
        if char_to_newchar[ind_ctm] >= len(context):
            continue
        newchar_to_matrix[char_to_newchar[ind_ctm]] = ctm
    for sc in sent_cls:
        sent_cls_n.append(char_to_newchar[sc])
    if is_training_global:
        start_position_n = char_to_newchar[start_position]
        if end_position == len(char_to_newchar):
            end_position_n = char_to_newchar[end_position - 1] + 1
        else:
            end_position_n = char_to_newchar[end_position]
    char_to_word_offset = []
    subwords_to_matrix = []
    doc_subwords = []
    sub_to_orig_index = []
    conlen = 0
    for indt, d_token in enumerate(doc_tokens):
        sub_tokens = tokenizer_global.tokenize(d_token)
        sum_toklen = 0
        unkc = 0
        conlen += len(d_token)
        for indst, sub_token in enumerate(sub_tokens):
            tok_len = len(sub_token)
            doc_subwords.append(sub_token)
            sub_to_orig_index.append(indt)
            if len(char_to_word_offset) < len(newchar_to_matrix):
                subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)])
            else:
                subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset) - 1])
            if sub_token.startswith('##'):
                tok_len -= 2
            elif sub_token.startswith("▁"):
                tok_len -= 1
            if sub_token == '<unk>':
                unkc += 1
                if len(sub_tokens) == indst + 1:
                    tok_len = len(d_token) - sum_toklen
                elif sub_tokens[indst + 1] == '<unk>':
                    tok_len = 1
                else:
                    tok_len = d_token.find(sub_tokens[indst + 1][0], sum_toklen) - sum_toklen
            prelen = len(char_to_word_offset)
            for rr in range(tok_len):
                if len(char_to_word_offset) < conlen:
                    char_to_word_offset.append(len(doc_subwords) - 1)
            sum_toklen += len(char_to_word_offset) - prelen
        if indt != len(doc_tokens) - 1:
            char_to_word_offset.append(len(doc_subwords))
            conlen += 1
        while len(char_to_word_offset) < conlen:
            char_to_word_offset.append(len(doc_subwords) - 1)
        while len(char_to_word_offset) > conlen:
            char_to_word_offset = char_to_word_offset[:-1]
        assert conlen == len(char_to_word_offset)
    assert len(char_to_word_offset) == len(context)

    sent_cls_w = []
    for sc in sent_cls_n:
        sent_cls_w.append(char_to_word_offset[sc])
    if is_training_global:
        start_position_w = char_to_word_offset[start_position_n]
        if end_position_n == len(char_to_word_offset):
            end_position_w = char_to_word_offset[end_position_n - 1] + 1
        else:
            end_position_w = char_to_word_offset[end_position_n]
    sent_cls_extend = []
    for ind_scw, scw in enumerate(sent_cls_w):
        doc_subwords.insert(scw + ind_scw, '<unk>')
        sub_to_orig_index.insert(scw + ind_scw, sub_to_orig_index[scw + ind_scw])
        subwords_to_matrix.insert(scw + ind_scw, subwords_to_matrix[scw + ind_scw])
        sent_cls_extend.append(scw + ind_scw)
        if is_training_global:
            if start_position_w >= scw + ind_scw:
                start_position_w += 1
            if end_position_w > scw + ind_scw:
                end_position_w += 1
    if is_training_global:
        if answer.lower() == 'yes':
            start_position_w = -1
            end_position_w = -1
        if answer.lower() == 'no':
            start_position_w = -2
            end_position_w = -2
    else:
        # 设置默认值
        start_position_w = None
        end_position_w = None
        sent_lbs = None
    example = HotpotQAExample(
        qas_id=qas_id,
        question_text=question,
        orig_tokens=doc_tokens,
        sub_to_orig_index=sub_to_orig_index,
        doc_tokens=doc_subwords,
        question_tokens=question_tokens,
        orig_answer_text=answer,
        start_position=start_position_w,
        end_position=end_position_w,
        sent_cls=sent_cls_extend,
        sent_lbs=sent_lbs,
        full_sents_mask=full_sents_mask,
        full_sents_lbs=full_sents_lbs,
        subwords_to_matrix=subwords_to_matrix
    )
    tmp_examples.append(example)
    return tmp_examples


def read_examples(input_file, supporting_para_file, tokenizer, is_training):
    global is_training_global
    global tokenizer_global
    global sp_dict_global

    datas = json.load(open(input_file, 'r', encoding='utf-8'))

    sp_dict = json.load(open(supporting_para_file, 'r'))
    for data in datas:
        get_id = data['_id']
        if 'level' in data and data['level'] == 'squad':
            sp_dict[get_id] = [0, ]

    examples = []
    no_answer_examples = 0
    is_training_global = is_training
    tokenizer_global = tokenizer
    sp_dict_global = sp_dict
    # for data_idx, data in enumerate(tqdm(datas)):
    pool_size = max(1, multiprocessing.cpu_count() // 4)
    pool = Pool(pool_size)
    for result in tqdm(pool.imap(func=process_single_data, iterable=datas),
                                              total=len(datas),
                                              desc="process examples..."):

       examples.extend(result)
    # for data in tqdm(datas):
    #     examples.extend(process_single_data(data=data))
    pool.close()
    pool.join()
    return examples
