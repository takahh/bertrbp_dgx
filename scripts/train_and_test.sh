#!/bin/bash

ORIG_PATH=/raid/t.kimura/data/datasets/
PYTHON_PATH=/home/kimura.t/bert-rbp/examples/run_finetune.py
KMER=3

###### args #######
RBP=$1
EPOCHS=$2
###################

MODEL_PATH=/raid/t.kimura/data/3-new-12w-0
# MODEL_PATH="ENTER THE PATH YOU SAVED DNABERT (such as 3-new-12w-0)"
echo "#######################"
echo $2
DATA_PATH=$ORIG_PATH$RBP/training_sample_finetune/
OUTPUT_PATH=$ORIG_PATH$RBP/finetuned_model/

echo "######### running training ##########"
python $PYTHON_PATH --overwrite_cache --model_type dna --tokenizer_name dna$KMER --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --output_dir $OUTPUT_PATH --do_train --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --learning_rate 2e-4 --num_train_epochs $EPOCHS --logging_steps 200 --warmup_percent 0.1 --hidden_dropout_prob 0.1 --overwrite_output_dir --weight_decay 0.01 --n_process 8

#python3 $PYTHON_PATH --model_type dna --tokenizer_name dna$KMER --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --output_dir $OUTPUT_PATH --do_train --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --learning_rate 2e-4 --num_train_epochs 3 --logging_steps 200 --warmup_percent 0.1 --hidden_dropout_prob 0.1 --overwrite_output_dir --weight_decay 0.01 --n_process 8

DATA_PATH=$ORIG_PATH$RBP/test_sample_finetune/
MODEL_PATH=$ORIG_PATH$RBP/finetuned_model/
PREDICT_PATH=$ORIG_PATH$RBP/finetuned_model/

python3 $PYTHON_PATH --model_type dna --local_rank 0 --tokenizer_name dna$KMER --model_name_or_path $MODEL_PATH --task_name dnaprom --do_eval --do_predict --data_dir $DATA_PATH --output_dir $MODEL_PATH --predict_dir $PREDICT_PATH --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --overwrite_output
