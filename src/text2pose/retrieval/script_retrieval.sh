#!/bin/bash -l
#SBATCH --job-name=examplejob       # Job name
#SBATCH --output=/scratch/project_465000903/LLava_video/posescript/log/examplejob.o%j     # Name of stdout output file
#SBATCH --error=/scratch/project_465000903/LLava_video/posescript/log/examplejob.e%j     # Name of stderr error file
#SBATCH --partition=standard-g      # Partition name (ensure this is correct!)
#SBATCH --nodes=1                   # Total number of nodes
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --gpus-per-node=1           # Allocate one GPU per MPI rank
#SBATCH --time=48:00:00              # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001867 # Project for billing
#SBATCH --mem=128G                   # Request 64GB of memory

source /scratch/project_465000903/posescript/bin/activate


action=$1 # (train|eval|demo)
checkpoint_type="best" # (last|best)

architecture_args=(
    --model PoseText
    --latentD 512
    --text_encoder_name 'distilbertUncased' --transformer_topping "avgp"
    # --text_encoder_name 'glovebigru_vocPSA2H2'
)

loss_args=(
    --retrieval_loss 'SigClip'
)

bonus_args=(
)

pretrained="ret_distilbert_dataPSA2" # used only if phase=='finetune'


##############################################################
# EXECUTE

# TRAIN
if [[ "$action" == *"train"* ]]; then

    phase=$2 # (pretrain|finetune)
    echo "NOTE: Expecting as argument the training phase. Got: $phase"
    seed=$3
    echo "NOTE: Expecting as argument the seed value. Got: $seed"
    
    # PRETRAIN 
    if [[ "$phase" == *"pretrain"* ]]; then
        srun python train_retrieval.py --dataset "posescript-A2" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --lr_scheduler "stepLR" --lr 0.0001 --lr_step 500 --lr_gamma 0.5 \
        --log_step 20 --val_every 10 \
        --batch_size 1024 --epochs 200 --seed $seed \
        >> /scratch/project_465000903/LLava_video/posescript/log/motionx_MyEncoder_whole_lr_0.0001_SigClip_log.log 2>&1

    # FINETUNE
    elif [[ "$phase" == *"finetune"* ]]; then

        python retrieval/train_retrieval.py --dataset "posescript-H2" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --apply_LR_augmentation \
        --lr_scheduler "stepLR" --lr 0.0002 --lr_step 40 --lr_gamma 0.5 \
        --batch_size 32 --epochs 200 --seed $seed \
        --pretrained $pretrained

    fi

fi


# EVAL QUANTITATIVELY
if [[ "$action" == *"eval"* ]]; then

    shift; experiments=( "$@" ) # gets all the arguments starting from the 2nd one

    for model_path in "${experiments[@]}"
    do
        echo $model_path
        python retrieval/evaluate_retrieval.py --dataset "posescript-H2" \
        --model_path ${model_path} --checkpoint $checkpoint_type \
        --split test
    done
fi


# EVAL QUALITATIVELY
if [[ "$action" == *"demo"* ]]; then

    experiment=$2 # only one at a time
    streamlit run retrieval/demo_retrieval.py -- --model_path $experiment --checkpoint $checkpoint_type

fi