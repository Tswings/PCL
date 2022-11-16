from __future__ import absolute_import, division, print_function
import os
import json

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer, RobertaTokenizer, ElectraTokenizer, AlbertTokenizer

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from collections import Counter
import string
import gc

from first_hop_type_data_helper import (HotpotQAExample,
                                        HotpotInputFeatures,
                                        read_hotpotqa_examples,
                                        convert_examples_to_features)
from first_type_predictor_config import get_config

sys.path.append("../pretrain_model")
from changed_model import ElectraForParagraphClassification


# 日志设置
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def write_predict_result(args, all_examples, all_features, all_results, is_training='train'):
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


def run_predict(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    cls_token = '[CLS]'
    sep_token = '[SEP]'
    unk_token = '[UNK]'
    pad_token = '[PAD]'
    tokenizer = ElectraTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)
    model = ElectraForParagraphClassification.from_pretrained(args.checkpoint_path)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # load dev_file, get dev_features
    dev_examples = read_hotpotqa_examples(input_file=args.dev_file, is_training='test')  #
    dev_features = convert_examples_to_features(
        examples=dev_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_training='dev',
        cls_token=cls_token,
        sep_token=sep_token,
        unk_token=unk_token,
        pad_token=pad_token
    )
    logger.info("dev feature num: {}".format(len(dev_features)))
    d_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    d_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    d_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    d_all_cls_mask = torch.tensor([f.cls_mask for f in dev_features], dtype=torch.long)
    d_all_cls_label = torch.tensor([f.cls_label for f in dev_features], dtype=torch.long)
    d_all_cls_weight = torch.tensor([f.cls_weight for f in dev_features], dtype=torch.float)
    d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
    dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                             d_all_cls_mask, d_all_cls_weight, d_all_example_index)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)

    # testing
    model.eval()
    all_results = []
    total_loss = 0
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "logit", "label"])

    with torch.no_grad():
        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            d_example_indices = d_batch[-1]
            if n_gpu == 1:
                d_batch = tuple(
                    t.squeeze(0).to(device) for t in d_batch)  # multi-gpu does scattering it-self
            inputs = {
                "input_ids": d_batch[0],
                "attention_mask": d_batch[1],
                "token_type_ids": d_batch[2],
                "cls_mask": d_batch[3],
                "cls_weight": d_batch[4],
            }
            dev_logits = model(**inputs)
            dev_logits = torch.sigmoid(dev_logits)

            for i, example_index in enumerate(d_example_indices):
                dev_logit = dev_logits[i].detach().cpu().tolist()
                dev_feature = dev_features[example_index.item()]
                dev_label = dev_logit.index(max(dev_logit))
                unique_id = dev_feature.unique_id
                all_results.append(RawResult(unique_id=unique_id,
                                             logit=dev_logit,
                                             label=dev_label))

    logger.info("start saving data...")
    logger.info("writing result to file...")
    if not os.path.exists(args.predict_result_path):
        logger.info("make new output dir:{}".format(args.predict_result_path))
    json.dump(all_results,
              open("{}/{}".format(args.predict_result_path, args.pred_file), "w", encoding="utf-8"))


if __name__ == '__main__':
    # --bert_model google/electra-large-discriminator --checkpoint_path ../../../question_type_model --model_name ElectraForParagraphClassification --dev_file ../../../data/input.json --pred_file pred.json --max_seq_length 150 --val_batch_size 4 --no_network True
    parser = get_config()
    args = parser.parse_args()
    run_predict(args=args)
