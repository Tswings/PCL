import argparse
import torch
import random
import numpy as np
import os
import gc
import sys
import logging
import pickle
import collections
from tqdm import trange, tqdm
from transformers import BertTokenizer, RobertaTokenizer, ElectraTokenizer, AlbertTokenizer
from torch.multiprocessing import Process
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from first_selector_config import get_config
from first_hop_type_data_helper import convert_examples_to_features, read_hotpotqa_examples
from first_hop_type_prediction_helper import prediction_evaluate, write_predictions

sys.path.append("../pretrain_model")
from changed_model import BElectraForParagraphClassification
from optimization import BertAdam, warmup_linear

logger = None


def logger_config(log_path, log_prefix='lwj', write2console=True):
    global logger
    logger = logging.getLogger(log_prefix)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if write2console:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
    logger.addHandler(handler)

    return logger


def dev_feature_getter(args,
                       tokenizer,
                       cls_token=None,
                       sep_token=None,
                       unk_token=None,
                       pad_token=None
                       ):
    dev_examples = read_hotpotqa_examples(args.dev_file, is_training='dev')
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    dev_feature_file = '{}/selector_1_dev_{}_{}_{}'.format(args.feature_cache_path,
                                                           list(filter(None, args.bert_model.split('/'))).pop(),
                                                           str(args.max_seq_length),
                                                           str(args.sent_overlap))
    if os.path.exists(dev_feature_file) and args.use_file_cache:
        with open(dev_feature_file, "rb") as dev_f:
            dev_features = pickle.load(dev_f)
    else:
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
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving dev features into cached file {}".format(dev_feature_file))
            with open(dev_feature_file, "wb") as writer:
                pickle.dump(dev_features, writer)
    logger.info("dev feature num: {}".format(len(dev_features)))
    d_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    d_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    d_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    d_all_cls_mask = torch.tensor([f.cls_mask for f in dev_features], dtype=torch.long)
    d_all_cls_label = torch.tensor([f.cls_label for f in dev_features], dtype=torch.long)
    d_all_cls_weight = torch.tensor([f.cls_weight for f in dev_features], dtype=torch.float)
    d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
    dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                             d_all_cls_mask, d_all_cls_label, d_all_cls_weight, d_all_example_index)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
    return dev_examples, dev_features, dev_dataloader


def convert_examples2file(args,
                          examples,
                          start_idxs,
                          end_idxs,
                          cached_train_features_file,
                          tokenizer,
                          cls_token=None,
                          sep_token=None,
                          unk_token=None,
                          pad_token=None
                          ):
    total_feature_num = 0
    for idx in range(len(start_idxs)):
        logger.info("start example idx: {} all num: {}".format(idx, len(start_idxs)))
        truly_train_examples = examples[start_idxs[idx]: end_idxs[idx]]
        new_train_cache_file = cached_train_features_file + '_' + str(idx)
        if os.path.exists(new_train_cache_file) and args.use_file_cache:
            with open(new_train_cache_file, "rb") as f:
                train_features = pickle.load(f)
        else:
            logger.info("convert {} example(s) to features...".format(len(truly_train_examples)))
            train_features = convert_examples_to_features(
                examples=truly_train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                is_training='train',
                cls_token=cls_token,
                sep_token=sep_token,
                unk_token=unk_token,
                pad_token=pad_token
            )
            logger.info("features gotten!")
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                new_tmp_train_cache_file = cached_train_features_file + '_' + str(idx)
                logger.info("  Saving train features into cached file {}".format(new_tmp_train_cache_file))
                logger.info("start saving features...")
                with open(new_tmp_train_cache_file, "wb") as writer:
                    pickle.dump(train_features, writer)
                logger.info("saving features done!")
        total_feature_num += len(train_features)
    logger.info('train feature_num: {}'.format(total_feature_num))
    return total_feature_num


def dev_evaluate(args,
                 model,
                 tokenizer,
                 n_gpu,
                 device,
                 model_name='BertForRelatedSentence',
                 step=0,
                 cls_token=None,
                 sep_token=None,
                 unk_token=None,
                 pad_token=None
                 ):
    dev_examples, dev_features, dev_dataloader = dev_feature_getter(args,
                                                                    tokenizer=tokenizer,
                                                                    cls_token=cls_token,
                                                                    sep_token=sep_token,
                                                                    unk_token=unk_token,
                                                                    pad_token=pad_token
                                                                    )
    model.eval()
    all_results = []
    total_loss = 0
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "logit"])
    has_sentence_result = True
    if model_name == 'BertForParagraphClassification' or 'ParagraphClassification' in model_name:
        has_sentence_result = False

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
                "cls_label": d_batch[4],
                "cls_weight": d_batch[5],
            }
            dev_loss, dev_logits = model(**inputs)
            dev_loss = torch.sum(dev_loss)
            dev_logits = torch.sigmoid(dev_logits)
            total_loss += dev_loss
            for i, example_index in enumerate(d_example_indices):
                dev_logit = dev_logits[i].detach().cpu().tolist()
                dev_feature = dev_features[example_index.item()]
                unique_id = dev_feature.unique_id
                all_results.append(RawResult(unique_id=unique_id,
                                             logit=dev_logit))

    acc, prec, em, rec = write_predictions(args=args,
                                           all_examples=dev_examples,
                                           all_features=dev_features,
                                           all_results=all_results,
                                           is_training='train',
                                           has_sentence_result=has_sentence_result,
                                           step=step)
    model.train()
    del dev_examples, dev_features, dev_dataloader
    gc.collect()
    return acc, prec, em, rec, total_loss


def train_iterator(args,
                   start_idxs,
                   cached_train_features_file,
                   tokenizer,
                   n_gpu,
                   model,
                   device,
                   optimizer,
                   num_train_optimization_steps,
                   steps_trained_in_current_epoch=0,
                   cls_token=None,
                   sep_token=None,
                   unk_token=None,
                   pad_token=None
                   ):
    global logger
    best_predict_acc = 0
    train_loss = 0
    global_steps = 0
    train_features = None
    for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        for start_idx in trange(len(start_idxs), desc="Split Data Index: {}".format(epoch_idx)):
            with open('{}_{}'.format(cached_train_features_file, str(start_idx)), "rb") as reader:
                train_features = pickle.load(reader)
            # 展开数据
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_cls_mask = torch.tensor([f.cls_mask for f in train_features], dtype=torch.long)
            # all_pq_end_pos = torch.tensor([f.pq_end_pos for f in train_features], dtype=torch.long)
            all_cls_label = torch.tensor([f.cls_label for f in train_features], dtype=torch.long)
            all_cls_weight = torch.tensor([f.cls_weight for f in train_features], dtype=torch.float)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                       all_cls_mask, all_cls_label, all_cls_weight)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            # 对每个Iteration进行模型训练
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration epoch:{}/{} data:{}/{}".format(
                    epoch_idx, int(args.num_train_epochs), start_idx, len(start_idxs)
            ))):
                if global_steps < steps_trained_in_current_epoch:
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        global_steps += 1
                    continue
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "cls_mask": batch[3],
                    "cls_label": batch[4],
                    "cls_weight": batch[5],
                }
                loss, _ = model(**inputs)
                if n_gpu > 1:
                    loss = loss.sum()  # mean() to average on multi-gpu.
                train_loss += loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if (global_steps + 1) % 100 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info(
                        "epoch:{:3d},data:{:3d},global_step:{:8d},loss:{:8.3f}".format(epoch_idx, start_idx,
                                                                                       global_steps, train_loss))
                    train_loss = 0
                # 固定步长验证结果并保存最佳结果
                if (global_steps + 1) % args.save_model_step == 0 and (
                        step + 1) % args.gradient_accumulation_steps == 0:
                    acc, prec, em, rec, total_loss = dev_evaluate(args=args,
                                                                  model=model,
                                                                  tokenizer=tokenizer,
                                                                  n_gpu=n_gpu,
                                                                  device=device,
                                                                  model_name=args.model_name,
                                                                  step=global_steps,
                                                                  cls_token=cls_token,
                                                                  sep_token=sep_token,
                                                                  unk_token=unk_token,
                                                                  pad_token=pad_token
                                                                  )
                    logger.info("epoch: {} data idx: {} step: {}".format(epoch_idx, start_idx, global_steps))
                    logger.info(
                        "acc: {} precision: {} em: {} recall: {} total loss: {}".format(acc, prec, em, rec, total_loss))

                    if acc > best_predict_acc:
                        best_predict_acc = acc
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        step_model_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_steps))
                        if not os.path.exists(step_model_dir):
                            os.mkdir(step_model_dir)
                        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
                        step_model_file = os.path.join(step_model_dir, 'pytorch_model.bin')
                        torch.save(model_to_save.state_dict(), output_model_file)
                        torch.save(model_to_save.state_dict(), step_model_file)
                        output_config_file = os.path.join(args.output_dir, 'config.json')
                        step_output_config_file = os.path.join(step_model_dir, 'config.json')
                        with open(step_output_config_file, "w") as f:
                            f.write(model_to_save.config.to_json_string())
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())
                        logger.info('saving model')

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_steps / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1
            del train_features, all_input_ids, all_input_mask, all_segment_ids, all_cls_label, all_cls_mask, all_cls_weight, train_data, train_dataloader
            gc.collect()
        acc, prec, em, rec, total_loss = dev_evaluate(args=args,
                                                      model=model,
                                                      tokenizer=tokenizer,
                                                      n_gpu=n_gpu,
                                                      device=device,
                                                      model_name=args.model_name,
                                                      step=global_steps,
                                                      cls_token=cls_token,
                                                      sep_token=sep_token,
                                                      unk_token=unk_token,
                                                      pad_token=pad_token
                                                      )
        logger.info("epoch: {} data idx: {} step: {}".format(epoch_idx, start_idx, global_steps))
        logger.info("acc: {} precision: {} em: {} recall: {} total loss: {}".format(acc, prec, em, rec, total_loss))

        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'pytorch_model_final.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, 'config.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        logger.info('saving model')

        if acc > best_predict_acc:
            best_predict_acc = acc
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Only save the model it-self
            step_model_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_steps))
            if not os.path.exists(step_model_dir):
                os.mkdir(step_model_dir)
            output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
            step_model_file = os.path.join(step_model_dir, 'pytorch_model.bin')
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(model_to_save.state_dict(), step_model_file)
            output_config_file = os.path.join(args.output_dir, 'config.json')
            step_output_config_file = os.path.join(step_model_dir, 'config.json')
            with open(step_output_config_file, "w") as f:
                f.write(model_to_save.config.to_json_string())
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            logger.info('saving model')


def run_train(rank=0, world_size=1):
    parser = get_config()
    args = parser.parse_args()
    if rank == 0 and not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, 'log_first_selector_{}_{}_{}_{}_{}_{}.log'.format(args.log_prefix,
                                                                                             args.bert_model.split('/')[
                                                                                                 -1],
                                                                                             args.output_dir.split('/')[
                                                                                                 -1],
                                                                                             args.train_batch_size,
                                                                                             args.max_seq_length,
                                                                                             args.doc_stride))
    logger = logger_config(log_path=log_path, log_prefix='')
    logger.info('-' * 15 + 'config' + '-' * 15)
    logger.info("All parameters are as follows：")
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))
    logger.info('-' * 30)
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
    #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    #
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if not args.train_file:
        raise ValueError('`train_file` is not specified!')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.over_write_result:
        raise ValueError('output_dir {} already exists!'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    unk_token = '[UNK]'
    pad_token = '[PAD]'

    tokenizer = ElectraTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BElectraForParagraphClassification.from_pretrained(args.bert_model)
    steps_trained_in_current_epoch = 0
    has_step = False

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model == DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # 将池化去除不进行更新
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    cached_train_features_file = "{}/selector_first_train_{}_{}_{}".format(args.feature_cache_path,
                                                                           list(filter(None, args.bert_model.split(
                                                                               '/'))).pop(),
                                                                           str(args.max_seq_length),
                                                                           str(args.sent_overlap))
    train_features = None
    model.train()
    train_examples = read_hotpotqa_examples(
        input_file=args.train_file,
        is_training='train')

    example_num = len(train_examples)
    logger.info('train example_num: {}'.format(example_num))
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    logger.info('{} examples and {} example file(s)'.format(example_num, len(start_idxs)))

    random.shuffle(train_examples)
    total_feature_num = convert_examples2file(args=args,
                                              examples=train_examples,
                                              start_idxs=start_idxs,
                                              end_idxs=end_idxs,
                                              cached_train_features_file=cached_train_features_file,
                                              tokenizer=tokenizer,
                                              cls_token=cls_token,
                                              sep_token=sep_token,
                                              unk_token=unk_token,
                                              pad_token=pad_token
                                              )
    #
    num_train_optimization_steps = int(
        total_feature_num / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    train_iterator(args=args,
                   start_idxs=start_idxs,
                   cached_train_features_file=cached_train_features_file,
                   tokenizer=tokenizer,
                   n_gpu=n_gpu,
                   model=model,
                   device=device,
                   optimizer=optimizer,
                   num_train_optimization_steps=num_train_optimization_steps,
                   steps_trained_in_current_epoch=steps_trained_in_current_epoch,
                   cls_token=cls_token,
                   sep_token=sep_token,
                   unk_token=unk_token,
                   pad_token=pad_token
                   )


if __name__ == '__main__':
    # --bert_model google/electra-large-discriminator --output_dir ../checkpoints/selector/first_hop_selector_type --feature_cache_path ../data/cache/selector/first_hop_selector_type --model_name ElectraForParagraphClassification --train_file ../../../data/hotpot_train_v1.1.json --dev_file ../../../data/hotpot_dev_distractor_v1.1.json --max_seq_length 50
    use_ddp = False
    if not use_ddp:
        run_train()
    else:
        world_size = 2
        processes = []
        for rank in range(world_size):
            p = Process(target=run_train, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
