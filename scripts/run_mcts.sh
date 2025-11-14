BIN_SIZE=0.1
C_PUCT=1.414
NUM_SIMULATION_STEPS=1667
BRANCH_K=3
TEMPERATURE_VALUES="[0.5,1.0,1.5]"
PREDIFFUSE_STEPS=300
EXPAND_STEPS=10
EVAL_FULL_MOL=True

# for C_PUCT in 1.0; do
HYDRA_FULL_ERROR=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python src/spec2mol_main.py \
    train.eval_batch_size=64 \
    dataset.max_count=null \
    general.test_only=/root/ms/DiffMS/checkpoints/diffms_canopus.ckpt \
    general.name=examine-mcts-evalfullmol-${EVAL_FULL_MOL}-c-puct-${C_PUCT}-bin-size-${BIN_SIZE}-branch-${BRANCH_K}-num-simulation-steps-${NUM_SIMULATION_STEPS}-prediffuse-steps-${PREDIFFUSE_STEPS}-expand-steps-${EXPAND_STEPS} \
    general.wandb_name=test \
    general.test_samples_to_generate=100 \
    general.val_samples_to_generate=100 \
    general.gpus=4 \
    general.seed=123 \
    mcts.use_mcts=true \
    mcts.num_simulation_steps=${NUM_SIMULATION_STEPS} \
    mcts.branch_k=${BRANCH_K} \
    mcts.c_puct=${C_PUCT} \
    mcts.time_budget_s=0.0 \
    mcts.verifier_batch_size=256 \
    mcts.verifier_type=graffms \
    mcts.num_workers=64 \
    mcts.bins_upper_mz=1500.0 \
    mcts.prediffuse_steps=${PREDIFFUSE_STEPS} \
    mcts.expand_steps=${EXPAND_STEPS} \
    mcts.similarity.bin_size=${BIN_SIZE} \
    mcts.debug_logging=true \
    mcts.eval_full_mol=${EVAL_FULL_MOL} \
    mcts.use_lmdb_cache=true \
    mcts.use_temperature=true \
    mcts.temperature_values=${TEMPERATURE_VALUES}
# done