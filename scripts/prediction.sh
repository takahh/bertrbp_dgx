#!/bin/bash

#ORIG_PATH=../sample_dataset/
#PYTHON_PATH=../examples/run_finetune.py
#KMER=3
#
#RBP=$1
## RBP=TIAL1
#
#DATA_PATH=$ORIG_PATH$RBP/test_sample_finetune/
#MODEL_PATH=$ORIG_PATH$RBP/finetuned_model/
#PREDICT_PATH=$ORIG_PATH$RBP/finetuned_model/

#python3 $PYTHON_PATH --model_type dna --tokenizer_name dna$KMER --model_name_or_path $MODEL_PATH --task_name dnaprom --do_predict --data_dir $DATA_PATH --output_dir $MODEL_PATH --predict_dir $PREDICT_PATH --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --overwrite_output

####################
#!/bin/bash

ORIG_PATH=/raid/t.kimura/data/datasets/
PYTHON_PATH=/home/kimura.t/bert-rbp/examples/run_finetune.py
KMER=3
RBP=$1
# RBP=TIAL1

MODEL_PATH=$2
# MODEL_PATH="ENTER THE PATH YOU SAVED DNABERT (such as 3-new-12w-0)"
echo "#######################"
echo $2
DATA_PATH=$ORIG_PATH$RBP/training_sample_finetune/
OUTPUT_PATH=$ORIG_PATH$RBP/finetuned_model/

DATA_PATH=$ORIG_PATH$RBP/test_sample_finetune/
MODEL_PATH=$ORIG_PATH$RBP/finetuned_model/
PREDICT_PATH=$ORIG_PATH$RBP/finetuned_model/

python3 $PYTHON_PATH --model_type dna --tokenizer_name dna$KMER --model_name_or_path $MODEL_PATH --task_name dnaprom --do_predict --data_dir $DATA_PATH --output_dir $MODEL_PATH --predict_dir $PREDICT_PATH --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --overwrite_output

