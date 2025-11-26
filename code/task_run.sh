# SG train old
export MODEL_NAME="sd2-community/stable-diffusion-2-1"
export OUTPUT_DIR="/Users/dimitraseferiadi/Documents/SG_Adapter/outputs/sg_adapter_run1"
export TRAIN_DIR="/Users/dimitraseferiadi/Documents/SG_Adapter/dataset/MultiRels"
export VALIDATION_FILE="/Users/dimitraseferiadi/Documents/SG_Adapter/dataset/MultiRels/valdata.jsonl"

accelerate launch train_sg_to_image_RAT.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DIR \
--resolution=768 --center_crop --random_flip \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--mixed_precision="no" \
--checkpointing_steps=20 \
--max_train_steps=2 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--caption_column "caption" \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--validation_file=${VALIDATION_FILE} \
--validation_steps=20 \
--val_num_images_per_condition=5 \
--seed 0 \
--output_dir=${OUTPUT_DIR} \
--use_sg_attn_mask \
--num_gnn_layers=1