export PYTHONNOUSERSITE=1

export HF_TOKEN="" #replace by your token
    
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

export CUDA_VISIBLE_DEVICES=0,1

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0

export TORCH_DEVICE="cuda:0"

# model_dirs: the path to the model you want to evaluate
# names: the name of your model
# tokenizer_dir: your model's tokenizer
# out_file: path to save the evaluation result

python distill_unlearn/eval_knowmem_muse.py \
    --model_dirs "" \
    --names "" \
    --tokenizer_dir "" \
    --corpus "books" \
    --metrics knowmem_f_muse \
    --out_file "" \
    --batch_size 8


