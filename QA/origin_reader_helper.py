import re
import sys
import bisect
import math
import json
import string
import collections
from collections import Counter

from transformers import BasicTokenizer


class HotpotQAExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 orig_tokens,
                 doc_tokens,
                 question_tokens,
                 sub_to_orig_index,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 sent_cls=None,
                 sent_lbs=None,
                 full_sents_mask=None,
                 full_sents_lbs=None,
                 mask_matrix=None,
                 subwords_to_matrix=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.orig_tokens = orig_tokens
        self.sub_to_orig_index = sub_to_orig_index
        self.doc_tokens = doc_tokens
        self.question_tokens = question_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.sent_cls = sent_cls
        self.sent_lbs = sent_lbs
        self.full_sents_mask = full_sents_mask
        self.full_sents_lbs = full_sents_lbs
        self.mask_matrix = mask_matrix
        self.subwords_to_matrix = subwords_to_matrix

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        qa_info = "qas_id:{} question:{}".format(self.qas_id, self.doc_tokens)
        if self.start_position:
            qa_info += " ,start position: {}".format(self.start_position)
            qa_info += " , end_position: {}".format(self.end_position)
        return qa_info


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 pq_end_pos=None,
                 start_position=None,
                 end_position=None,
                 sent_mask=None,
                 sent_lbs=None,
                 sent_weight=None,
                 mask=None,
                 content_len=None,
                 word_sim_matrix=None,
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pq_end_pos = pq_end_pos
        self.start_position = start_position
        self.end_position = end_position
        self.sent_mask = sent_mask
        self.sent_lbs = sent_lbs
        self.sent_weight = sent_weight
        self.mask = mask
        self.content_len = content_len
        self.word_sim_matrix = word_sim_matrix


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx - 1] - target) if test_func(a[idx - 1]) else 1e200
        if d1 > d2:
            return a[idx - 1], d2
        else:
            return a[idx], d1


def fix_span(para, offsets, span):
    span = span.strip()
    parastr = " ".join(para)

    assert span in parastr, '{}\t{}'.format(span, parastr)
    # print([y for x in offsets for y in x])
    begins = []
    ends = []
    for o in offsets:
        begins.append(o[0])
        ends.append(o[1])
    # begins, ends = map(list, zip([y for x in offsets for y in x]))#在列表前加*号，会将列表拆分成一个一个的独立元素

    best_dist = 1e200
    best_indices = None

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()
        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < begin_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > end_offset)

        if d1 + d2 < best_dist:
            best_dist = d1 + d2
            best_indices = (fixed_begin, fixed_end)
            if best_dist == 0:
                break

    assert best_indices is not None
    return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # if verbose_logging:
        #     logger.info(
        #         "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        # if verbose_logging:
        #     logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
        #                 orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        # if verbose_logging:
        #     logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        # if verbose_logging:
        #     logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def is_whitespace(ch):
    if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F or ch == '\xa0':
        return True
    return False


def write_predictions(tokenizer, all_examples, all_features, all_results, n_best_size=20,
                      max_answer_length=20, do_lower_case=True):
    """Write final predictions to the json file and log-odds of null if needed."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    sp_preds = {}
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        sent_pred_logit = [0.0] * len(example.doc_tokens)
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logit, n_best_size)
            end_indexes = _get_best_indexes(result.end_logit, n_best_size)
            for ind_rsl, rsl in enumerate(result.sent_logit):
                if feature.sent_mask[ind_rsl] == 1 and feature.token_is_max_context.get(ind_rsl, False):
                    sent_pred_logit[feature.token_to_orig_map[ind_rsl]] = rsl
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    # start_index+=offset
                    # end_index+=offset
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logit[start_index],
                            end_logit=result.end_logit[end_index]))
        sent_pred_logit = [spl for ind_spl, spl in enumerate(sent_pred_logit) if ind_spl in example.sent_cls]
        sp_pred = []
        pointer = 0
        for fsm in example.full_sents_mask:
            if fsm == 0:
                sp_pred.append(0.0)
            else:
                sp_pred.append(sent_pred_logit[pointer])
                pointer += 1
        sp_preds[example.qas_id] = sp_pred
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["start", "end", "text", "start_logit", 'end_logit'])

        seen_predictions = {}
        nbest = []
        def _token_helper(input_tokens):
            output_tokens = []
            for input_token in input_tokens:
                if len(output_tokens) == 0 or input_token.startswith("▁"):
                    output_tokens.append(input_token.lstrip("▁"))
                else:
                    output_tokens[-1] += input_token
            return output_tokens
        is_albert = False
        for token in feature.tokens:
            if "▁" in token:
                is_albert = True
                break
        for pred in prelim_predictions:
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                tok_tokens = [tt for tt in tok_tokens if tt != '<unk>']
                if is_albert:
                    tok_tokens = _token_helper(tok_tokens)
                orig_doc_start = example.sub_to_orig_index[feature.token_to_orig_map[pred.start_index]]
                orig_doc_end = example.sub_to_orig_index[feature.token_to_orig_map[pred.end_index]]
                orig_tokens = example.orig_tokens[orig_doc_start:(orig_doc_end + 1)]
                orig_tokens = [ot for ot in orig_tokens if ot != '<unk>']

                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.replace("▁", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                # tok_text=tokenizer.convert_tokens_to_string(tok_tokens)

                orig_text = " ".join(orig_tokens)
                # orig_text = orig_text.replace("##", "").strip()
                # orig_text = orig_text.strip()
                # orig_text = " ".join(orig_text.split())
                # orig_text=tokenizer.convert_tokens_to_string(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, False)
                tok_text_f = ''.join(tok_text.split())
                final_text_f = ''.join(final_text.split()).lower()
                start_offset = final_text_f.find(tok_text_f)
                end_offset = len(final_text_f) - start_offset - len(tok_text_f)
                if start_offset >= 0:
                    if end_offset != 0:
                        final_text = final_text[start_offset:-end_offset]
                    else:
                        final_text = final_text[start_offset:]
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    start=orig_doc_start,
                    end=orig_doc_end,
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        # if not nbest:
        #     nbest.append(
        #         _NbestPrediction(start=0, end=0, text="", logit=-1e6))
        # nbest.append(_NbestPrediction(start=0, end=0, text="", logit=result.start_logit[0]+result.end_logit[0]))
        nbest.append(_NbestPrediction(start=1, end=1, text="yes", start_logit=result.start_logit[1],
                                      end_logit=result.end_logit[1]))
        nbest.append(_NbestPrediction(start=2, end=2, text="no", start_logit=result.start_logit[2],
                                      end_logit=result.end_logit[2]))
        nbest = sorted(nbest, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        # assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        all_predictions[example.qas_id] = nbest_json[0]

    return nbest_json, all_predictions, sp_preds


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(prediction, gold):
    tp, fp, fn = 0, 0, 0
    for p, g in zip(prediction, gold):
        # if p==0.0:
        #     if g==1:
        #         fn+=1
        # else:
        #     if p[0]<p[1] and g==1:
        #         tp+=1
        #     if p[0]<p[1] and g==0:
        #         fp+=1
        #     if p[0]>p[1] and g==1:
        #         fn+=1
        if p > 0.5 and g == 1:
            tp += 1
        if p > 0.5 and g == 0:
            fp += 1
        if p <= 0.5 and g == 1:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return f1, em, prec, recall


def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N
    print(metrics)


def evaluate(eval_examples, answer_dict, sp_preds):
    ans_f1 = ans_em = sp_f1 = sp_em = joint_f1 = joint_em = total = 0
    for ee in eval_examples:
        pred = answer_dict[ee.qas_id]['text']
        ans = ee.orig_answer_text
        total += 1
        # print(pred)
        # print(ans)
        # print()
        a_f1, a_prec, a_recall = f1_score(pred, ans)
        ans_f1 += a_f1
        a_em = exact_match_score(pred, ans)
        ans_em += a_em
        s_f1, s_em, s_prec, s_recall = update_sp(sp_preds[ee.qas_id], ee.full_sents_lbs)
        sp_f1 += s_f1
        sp_em += s_em
        j_prec = a_prec * s_prec
        j_recall = a_recall * s_recall
        if j_prec + j_recall > 0:
            j_f1 = 2 * j_prec * j_recall / (j_prec + j_recall)
        else:
            j_f1 = 0.
        j_em = a_em * s_em
        joint_f1 += j_f1
        joint_em += j_em
    return ans_f1 / total, ans_em / total, sp_f1 / total, sp_em / total, joint_f1 / total, joint_em / total
