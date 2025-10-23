cd contriever

python finetuning.py \
    --model_path facebook/mcontriever \
    --train_data ../data/contriever_training_data.jsonl \
    --eval_data ../data/contriever_validation_data.jsonl \
    --chunk_length 512 \
    --seed 42 \
    --total_steps 2000 --save_freq 100 \

