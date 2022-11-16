import json
import os
import collections


def prediction_evaluate(args,
                        paragraph_results,
                        labels,
                        thread=0.5,
                        step=0):
    """  """
    true_dev_dict = json.load(open(args.dev_file, "r"))
    all_dev_related_paragraph_dict = {}
    for info in true_dev_dict:
        get_id = info['_id']
        q_type = info['type']
        true_values = 0 if q_type == 'comparison' else 1
        all_dev_related_paragraph_dict[get_id] = true_values
    new_para_result = {}
    for k, v in paragraph_results.items():
        new_para_result[k] = v
    predict_dict = {}
    for k, v in new_para_result.items():
        max_value = max(v)
        max_idx = 0
        for pre_idx, pre_v in enumerate(v):
            if pre_v == max_value:
                max_idx = pre_idx
                break
        predict_dict[k] = max_idx
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "predict_all_dev_result_{}.json".format(step)), "w") as writer:
        json.dump(new_para_result, writer)
    with open(os.path.join(args.output_dir, "predict_dev_result_{}.json".format(step)), "w") as writer:
        json.dump(predict_dict, writer)
    true_num = 0
    bad_num = 0
    for k, v in predict_dict.items():
        if v == all_dev_related_paragraph_dict[k]:
            true_num += 1
        else:
            bad_num += 1
    acc = 1.0 * true_num / len(predict_dict)
    return acc, acc, acc, acc


def prediction_evaluate_tmp(args,
                        paragraph_results,
                        labels,
                        thread=0.5):
    """  """
    p_recall = p_precision = sent_em = sent_acc = sent_recall = 0
    all_count = 0
    new_para_result = {}
    for k, v in paragraph_results.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        if q_id not in new_para_result:
            new_para_result[q_id] = [[0] * 10, [0] * 10]
        new_para_result[q_id][0][context_id] = v
    for k, v in labels.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        new_para_result[q_id][1][context_id] = v[0]
    for k, v in new_para_result.items():
        all_count += 1
        p11 = p10 = p01 = p00 = 0
        max_v = max(v[0])
        min_v = min(v[0])
        max_logit = -100
        max_result = False
        # TODO: check v format
        for idx, (paragraph_result, label) in enumerate(zip(v[0], v[1])):
            if paragraph_result > max_logit:
                max_logit = paragraph_result
                max_result = True if label == 1 else max_result
            # MinMax Scaling
            paragraph_result = (paragraph_result - min_v) / (max_v - min_v)
            paragraph_result = 1 if paragraph_result > thread else 0
            if paragraph_result == 1 and label == 1:
                p11 += 1
            elif paragraph_result == 1 and label == 0:
                p10 += 1
            elif paragraph_result == 0 and label == 1:
                p01 += 1
            elif paragraph_result == 0 and label == 0:
                p00 += 1
            else:
                raise NotImplemented
        if p11 + p01 != 0:
            p_recall += p11 / (p11 + p01)
        else:
            print("error in calculate paragraph recall!")
        if p11 + p10 != 0:
            p_precision += p11 / (p11 + p10)
        else:
            print("error in calculate paragraph precision!")
        if p11 == 2 and p10 == 0:
            sent_em += 1
        if p01 == 0:
            sent_recall += 1
        if max_result:
            sent_acc += 1
    return sent_acc / all_count, p_precision / all_count, sent_em / all_count, sent_recall / all_count


def write_predictions(args, all_examples, all_features, all_results, is_training='train', has_sentence_result=True, step=0):
    example_index2features = collections.defaultdict(list)
    for feature in all_features:
        example_index2features[feature.example_index].append(feature)
    unique_id2result = {x[0]: x for x in all_results}
    paragraph_results = {}
    labels = {}
    for example_index, example in enumerate(all_examples):
        features = example_index2features[example_index]
        id = features[0].unique_id
        get_feature = features[0]
        get_feature_id = get_feature.unique_id
        raw_result = unique_id2result[id].logit
        paragraph_results[id] = raw_result

    if is_training == 'test':
        return 0, 0, 0, 0
    else:
        return prediction_evaluate(args=args,
                                   paragraph_results=paragraph_results,
                                   labels=labels,
                                   step=step)
