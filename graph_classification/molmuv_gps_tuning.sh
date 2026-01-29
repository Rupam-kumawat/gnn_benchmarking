#!/usr/bin/env bash
set -e

############################################
# GPU PINNING
############################################
export CUDA_VISIBLE_DEVICES=3

############################################
# CONFIG
############################################
SCRIPT="main_molmuv_gps.py"
DATASET="ogbg-molmuv"
EPOCHS=50
BATCH_SIZE=32
RUNS=3

############################################
# GRID PARAMETERS
############################################
LAYERS=(1 3 5 7 10)
HIDDEN_DIMS=(64 128 256 512)
POOLING=("mean" "add")

# Loss priority: focal → standard
LOSSES=("focal" "standard")

# Focal loss defaults (DO NOT GRID)
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0

############################################
# LOGGING
############################################
LOG_DIR="grid_logs"
mkdir -p ${LOG_DIR}

############################################
# GRID SEARCH
############################################
for L in "${LAYERS[@]}"; do
  for H in "${HIDDEN_DIMS[@]}"; do
    for POOL in "${POOLING[@]}"; do
      for LOSS in "${LOSSES[@]}"; do

        if [[ "${LOSS}" == "focal" ]]; then
          EXP_NAME="L${L}_H${H}_${POOL}_focal"
          echo ">>> Running ${EXP_NAME} on GPU 3"

          python ${SCRIPT} \
            --dataset ${DATASET} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --num_layers ${L} \
            --channels ${H} \
            --pool ${POOL} \
            --use_focal_loss \
            --focal_alpha ${FOCAL_ALPHA} \
            --focal_gamma ${FOCAL_GAMMA} \
            --runs ${RUNS} \
            > ${LOG_DIR}/${EXP_NAME}.log 2>&1

        else
          EXP_NAME="L${L}_H${H}_${POOL}_standard"
          echo ">>> Running ${EXP_NAME} on GPU 3"

          python ${SCRIPT} \
            --dataset ${DATASET} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --num_layers ${L} \
            --channels ${H} \
            --pool ${POOL} \
            --runs ${RUNS} \
            > ${LOG_DIR}/${EXP_NAME}.log 2>&1
        fi

      done
    done
  done
done

echo "✅ Grid search completed on GPU 3."
