import argparse


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def get_config():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model", default='roberta-large', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default='../../data/checkpoints/qa_base_20211022_with_entity_wo_question_entity_dim_10_wi_context_mask', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--model_name", type=str, default='RobertaForQuestionAnsweringForwardBest',
                        help="must be BertForQuestionAnsweringCoAttention"
                             "\BertForQuestionAnsweringThreeCoAttention"
                             "\BertForQuestionAnsweringThreeSameCoAttention"
                             "\BertForQuestionAnsweringForward")
    parser.add_argument("--train_file", default='../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    # parser.add_argument("--overwrite_result", dest='overwrite_result', action='store_true')
    # parser.add_argument("--no-over_write_result", dest='overwrite_result', action='store_false')
    parser.add_argument("--overwrite_result", type=str2bool, default=True)
    parser.add_argument("--log_prefix", default="qa_base_20211022_with_entity_test", type=str)
    parser.add_argument("--log_path", default="../../log", type=str)
    parser.add_argument("--dev_file", default='../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_supporting_para_file",
                        default='../../data/selector/second_hop_related_paragraph_result/train_related.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_supporting_para_file",
                        default='../../data/selector/second_hop_related_paragraph_result/dev_related.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--feature_cache_path", default='../../data/cache/test', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--feature_suffix", default="origin_model", type=str,
                        help="cache feature suffix")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=256, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        type=str2bool, default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--save_model_step',
                        type=int, default=1000,
                        help="The proportion of the validation set")

    return parser
