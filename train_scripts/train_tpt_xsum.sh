python3 finetune.py \
--dataset_name=xsum \
--cache_dir=cache \
--data_dir=data \
--tokenizer_name=t5-small \
--model_name_or_path=tpt-small-discrete \
--learning_rate=5e-3 \
--max_source_length=512 \
--max_target_length=64 \
--gradient_accumulation_steps=1 \
--train_batch_size=16 \
--num_train_epochs=100 \
--eval_batch_size=32 \
--n_gpu=4 \
--run_id=00 \
--train_from_scratch \
--scheduler_type=inverse_sqrt \
--output_dir=out \
--overwrite_output_dir \
--label_smooth=0.1 \
--optim_name=adafactor \
--fix_lr_step=30000 \
--num_roles_per_layer=50 \
--do_train

## Optional
#--role_regularization=0.1
#--resume_from_epoch=49 \
#--resume_ckpt_path=$PT_MAP_OUTPUT_DIR \
# --seed=10 \