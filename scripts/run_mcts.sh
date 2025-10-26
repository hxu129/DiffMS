HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=3 python src/spec2mol_main.py \
    train.eval_batch_size=40 \
    dataset.max_count=null \
    general.test_only=/local3/ericjiang/wgc/huaxu/ms/DiffMS/checkpoints/diffms_canopus.ckpt \
    general.name=dev \
    general.wandb_name=test \
    general.test_samples_to_generate=100 \
    general.gpus=1 \
    general.seed=123 \
    mcts.use_mcts=true \
    mcts.num_simulation_steps=1000 \
    mcts.branch_k=2 \
    mcts.c_puct=0.05 \
    mcts.time_budget_s=0.0 \
    mcts.verifier_batch_size=32 \
    mcts.verifier_type=iceberg \
    mcts.t_thresh=200
