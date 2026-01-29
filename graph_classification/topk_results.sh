#!/bin/bash

# ==========================================================
# Full Pipeline: Evaluation and Rebuttal Report Generation
# ==========================================================

DATASETS=("COX2" "ogbg-molhiv" "AIDS")
GNN_MODELS=("gcn" "gat" "sage")
# GPS_MODELS=("gps" "subgraphormer")

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "PROCESSING DATASET: $DATASET"
    echo "=========================================================="

    # Determine dataset type
    if [[ "$DATASET" == "ogbg-molhiv" ]]; then
        TYPE="ogb"
    else
        TYPE="tu"
    fi

    # 1. Run Top-K Evaluation for GCN, GAT, SAGE
    for MODEL in "${GNN_MODELS[@]}"; do
        echo "[GNN Eval] Running $MODEL on $DATASET..."
        python analyze_topk_results_molhiv.py \
            --dataset "$DATASET" \
            --model_name "$MODEL" \
            --device 0
    done

    # 2. Run Top-K Evaluation for GPS and Subgraphormer
    for MODEL in "${GPS_MODELS[@]}"; do
        echo "[GPS/GT Eval] Running $MODEL on $DATASET..."
        python analyze_topk_results_molhiv_gps.py \
            --dataset "$DATASET" \
            --model_name "$MODEL" \
            --dataset_type "$TYPE" \
            --device 0
    done

    # 3. Generate Final Rebuttal Report
    # Using the filter lr-0.0001 as requested for GNN results
    echo "[Report] Generating aggregated results for $DATASET..."
    python molhiv_rebuttal.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --filter "lr-0.0001"

    echo "Finished $DATASET."
    echo ""
done

echo "✅ All benchmarks and reports are complete."