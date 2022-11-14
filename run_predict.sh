
echo "Select relevant paragraphs"

echo "PS Step1:"
cd PS1
CUDA_VISIBLE_DEVICES=0,5 python run_ps1.py --bert_model google/electra-large-discriminator --checkpoint_path ps1_model --model_name ElectraForParagraphClassification --dev_file ../data/input.json --best_paragraph_file dev_best_paragraph.json --predict_result_path ps1_result --related_paragraph_file dev_related_paragraph.json --new_context_file dev_new_context.json --val_batch_size 32 --no_network True
echo "Step1 finished!"
cd ..

cd PS2
echo "PS Step2:"
CUDA_VISIBLE_DEVICES=0,5 python run_ps2.py --bert_model google/electra-large-discriminator --checkpoint_path ps2_model --model_name ElectraForParagraphClassification --dev_file ../data/input.json --first_predict_result_path ../PS1/ps1_result --second_predict_result_path ps2_result --final_related_result dev_related.json --best_paragraph_file dev_best_paragraph.json --related_paragraph_file dev_related_paragraph.json --new_context_file dev_new_context.json --val_batch_size 32 --no_network True
echo "Step2 finished!"
cd ..

cd QA
echo "Question Answering:"
CUDA_VISIBLE_DEVICES=0,5 python run_qa.py --bert_model albert-xxlarge-v2 --checkpoint_dir qa_model --output_dir qa_results --predict_file pred.json --test_supporting_para_file ../PS2/ps2_result/dev_related.json --model_name AlbertForQuestionAnswering --log_prefix predict_log --overwrite_result True --test_file ../data/input.json --local_rank -1 --val_batch_size 32 --no_network True

echo "Good"










