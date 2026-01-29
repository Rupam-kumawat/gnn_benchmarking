#!/bin/bash

# ====================================================================================
#  Unified Graph Classification Experiment Runner
#
# This script combines all experiments and strictly follows the hyperparameter
# specifications provided in the image for MPNNs, GraphGPS, and Subgraphormer.
# ====================================================================================

# --- Configuration ---
# Use the first command-line argument as the GPU ID, or default to 0
GPU_ID=${1:-0}
echo "Running on GPU: $GPU_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ====================================================================================
#  SECTION 1: Standard GNN Experiments (GCN, GAT, GraphSAGE)
# ====================================================================================
echo "#################################################################"
echo "### Starting Standard GNN Experiments (GCN, GAT, SAGE)        ###"
echo "#################################################################"

GNN_DATASETS=("ogbg-molhiv" "COX2" "COLLAB")
GNN_MODELS=("gcn" "gat" "sage")
GNN_LR=0.001
GNN_HIDDEN_DIM=128
GNN_DROPOUT=0.2
GNN_LAYERS=(1 3 4 5 7) # As per image
GNN_POOLING=("mean" "add")

for dataset in "${GNN_DATASETS[@]}"; do
    # --- Set conditional parameters based on the image ---
    if [ "$dataset" == "ogbg-molhiv" ]; then
      DATASET_TYPE="ogb"
      EPOCHS=100
    else # COLLAB and COX2
      DATASET_TYPE="tu"
      EPOCHS=300
    fi

    for model in "${GNN_MODELS[@]}"; do
        for layers in "${GNN_LAYERS[@]}"; do
            for pool in "${GNN_POOLING[@]}"; do
                echo "----------------------------------------------------------------"
                echo "RUNNING GNN:"
                echo "  Dataset: $dataset | Model: $model | Pool: $pool"
                echo "  Params: lr=$GNN_LR, hidden=$GNN_HIDDEN_DIM, dropout=$GNN_DROPOUT, layers=$layers, epochs=$EPOCHS"
                echo "----------------------------------------------------------------"

                python graph_classification/main_graph_gnn.py \
                  --dataset_type "$DATASET_TYPE" \
                  --dataset "$dataset" \
                  --model_name "$model" \
                  --pool "$pool" \
                  --lr "$GNN_LR" \
                  --hidden_channels "$GNN_HIDDEN_DIM" \
                  --dropout "$GNN_DROPOUT" \
                  --num_layers "$layers" \
                  --device "$GPU_ID" \
                  --epochs "$EPOCHS" \
                  --metric "acc" \
                  --use_bn \
                  --use_residual \
                  --runs 3
            done
        done
    done
done


# ====================================================================================
#  SECTION 2: GraphGPS Model Experiments
# ====================================================================================
echo "#################################################################"
echo "### Starting GraphGPS Model Experiments                       ###"
echo "#################################################################"

GPS_DATASETS=("ogbg-molhiv" "COX2" "COLLAB")
GPS_HIDDEN_DIM=64
GPS_DROPOUT=0.2
GPS_HEADS=4
GPS_EPOCHS=100
GPS_POOLING=("mean" "add") # Script uses "add" for sum pooling
GPS_RUNS=3

for dataset in "${GPS_DATASETS[@]}"; do
    # --- Set conditional parameters based on the image ---
    if [ "$dataset" == "COLLAB" ]; then
        PYTHON_SCRIPT="graph_classification/GPS_on_collab.py"
        CURRENT_GPS_LAYERS=(1 5)
    else # ogbg-molhiv and COX2
        PYTHON_SCRIPT="graph_classification/gps_molhiv_cox2_datasets.py"
        if [ "$dataset" == "ogbg-molhiv" ]; then
            CURRENT_GPS_LAYERS=(1 5 10)
        else # COX2
            CURRENT_GPS_LAYERS=(1 5)
        fi
    fi

    for layers in "${CURRENT_GPS_LAYERS[@]}"; do
        for pool in "${GPS_POOLING[@]}"; do
            echo "----------------------------------------------------------------"
            echo "RUNNING GPS:"
            echo "  Dataset: $dataset | Layers: $layers | Pool: $pool"
            echo "  Script: $PYTHON_SCRIPT"
            echo "----------------------------------------------------------------"

            python "$PYTHON_SCRIPT" \
                --dataset "$dataset" \
                --epochs "$GPS_EPOCHS" \
                --batch_size 32 \
                --lr 0.0001 \
                --channels "$GPS_HIDDEN_DIM" \
                --num_layers "$layers" \
                --runs "$GPS_RUNS" \
                --metric "acc" \
                --conv_type gated \
                --pool "$pool"
        done
    done
done


# ====================================================================================
#  SECTION 3: Subgraphormer Experiments
# ====================================================================================
echo "#################################################################"
echo "### Starting Subgraphormer Experiments                        ###"
echo "#################################################################"
# NOTE: COLLAB is skipped as per the image due to Out of Memory errors.

SUB_DATASETS=("ogbg-molhiv" "COX2")
SUB_HIDDEN_DIM=128  # As per image for molhiv and COX2
SUB_LAYERS=5        # As per image for molhiv and COX2
SUB_POOLING=("mean" "add")
SUB_DROPOUT=0.2
SUB_HEADS=4
SUB_EPOCHS=100
SUB_LR=0.001

for dataset in "${SUB_DATASETS[@]}"; do
    if [ "$dataset" == "ogbg-molhiv" ]; then
      DATASET_TYPE="ogb"
    else
      DATASET_TYPE="tu"
    fi

    for pool in "${SUB_POOLING[@]}"; do
        echo "----------------------------------------------------------------"
        echo "RUNNING Subgraphormer:"
        echo "  Dataset: $dataset | Pool: $pool"
        echo "  Params: lr=$SUB_LR, hidden=$SUB_HIDDEN_DIM, dropout=$SUB_DROPOUT, layers=$SUB_LAYERS, heads=$SUB_HEADS"
        echo "----------------------------------------------------------------"

        python graph_classification/main_graphclass.py \
          --dataset_type "$DATASET_TYPE" \
          --dataset "$dataset" \
          --model_family "gt" \
          --model_name "subgraphormer" \
          --metric "acc" \
          --pool "$pool" \
          --lr "$SUB_LR" \
          --hidden_channels "$SUB_HIDDEN_DIM" \
          --dropout "$SUB_DROPOUT" \
          --num_layers "$SUB_LAYERS" \
          --nhead "$SUB_HEADS" \
          --device "$GPU_ID" \
          --epochs "$SUB_EPOCHS" \
          --runs 3
    done
done

echo "================================================================"
echo "All graph classification experiments finished."
echo "================================================================"