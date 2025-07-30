# SG train old
export MODEL_NAME=""
export OUTPUT_DIR=""
export TRAIN_DIR=""
export VALIDATION_FILE=""

accelerate launch train_sg_to_image_RAT.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DIR \
--resolution=768 --center_crop --random_flip \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--mixed_precision="fp16" \
--checkpointing_steps=2000 \
--max_train_steps=30000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--caption_column "caption" \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--validation_file=${VALIDATION_FILE} \
--validation_steps=2000 \
--val_num_images_per_condition=5 \
--seed 0 \
--output_dir=${OUTPUT_DIR} \
--use_sg_attn_mask