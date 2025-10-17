python src/spec2mol_main.py \
    dataset.max_count=100 \
    train.eval_batch_size=128 \
    general.wandb_name=test \
    general.test_only=/root/ms/DiffMS/checkpoints/diffms_canopus.ckpt \
    general.name=dev \
    general.sample_every_val=1000 \
    general.test_samples_to_generate=10 \
    general.val_samples_to_generate=100 \
    general.log_every_steps=50 \
    general.gpus=1 \
    general.seed=123 \
    mcts.use_mcts=false