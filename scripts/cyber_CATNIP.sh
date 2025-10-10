export HF_TOKEN="" #replace by your token

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# replace by your WANDB token and PROJECT NAME
export WANDB_API_KEY=""
export WANDB_PROJECT=""          
export WANDB_WATCH="false"
export WANDB_NAME=""        

#replace with your environment
export TRANSFORMERS_CACHE=""
export XDG_CACHE_HOME=""
export HF_HOME=""
export HF_DATASETS_CACHE=""
export HUGGINGFACE_HUB_CACHE=""


mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HUGGINGFACE_HUB_CACHE

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"
FORGET_FILE="data/cyber-forget-corpus.jsonl"

BATCH_SIZE=2  
NUM_EPOCHS=1.8
LEARNING_RATE=5e-6
beta=2
TEMPERATURE=1.0  
MAX_LENGTH=512

OUTPUT_DIR="output_models/cyber_CATNIP_maxlen${MAX_LENGTH}_beta_${beta}_lr_${LEARNING_RATE}_epoch_${NUM_EPOCHS}"

OUT_LOG="cyber_CATNIP_maxlen${MAX_LENGTH}_beta_${beta}_lr_${LEARNING_RATE}_epoch_${NUM_EPOCHS}.out"
ERR_LOG="cyber_CATNIP_maxlen${MAX_LENGTH}_beta_${beta}_lr_${LEARNING_RATE}_epoch_${NUM_EPOCHS}.err"


echo "================================"
echo "Model: $MODEL_NAME"
echo "Forget Data: $FORGET_FILE"
echo "Retain Data: $RETAIN_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Temperature: $TEMPERATURE"
echo "Max Length: $MAX_LENGTH"
echo "beta: $beta"
echo "================================"

torchrun --nproc_per_node=2 WMDP_cyber_CATNIP.py \
    --model_name $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --forget_file $FORGET_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --max_length $MAX_LENGTH \
    --beta $beta \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "cyber_CATNIP_maxlen${MAX_LENGTH}_beta_${beta}_lr_${LEARNING_RATE}_epoch_${NUM_EPOCHS}" \
    1>${OUT_LOG} 2>${ERR_LOG}

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR" 