#!/bin/bash

# ==================================================================================
# Amazon-Ratings Ordinal Classification - Hyperparameter Sweep (GPU 0)
# ==================================================================================

# GPU Configuration
GPU=3
DATASET="amazon-ratings"
RUNS=10
EPOCHS=2500
LR=0.001

# Create results directory
mkdir -p results/${DATASET}

# Log file
LOG_FILE="logs/${DATASET}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "========================================" | tee -a ${LOG_FILE}
echo "Starting Amazon-Ratings Ordinal Sweep" | tee -a ${LOG_FILE}
echo "GPU: ${GPU}" | tee -a ${LOG_FILE}
echo "Date: $(date)" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# Counter for tracking progress
total_experiments=0
completed_experiments=0

# Count total experiments
for hidden in  256 ; do
    for dropout in 0.5 0.7; do
        for layers in 1 3 5 7 10; do
            for gnn in gcn gat sage; do
                for metric in acc qwk; do
                    for loss in ce dwce; do
                        if [ "$loss" = "dwce" ]; then
                            for alpha in 0.1 0.5; do
                                ((total_experiments++))
                            done
                        else
                            ((total_experiments++))
                        fi
                    done
                done
            done
        done
    done
done

echo "Total experiments to run: ${total_experiments}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Outer loops: hidden dimensions and local layers
for hidden in 256 ; do
    for layers in 1 3 5 7 10; do
        for dropout in 0.5 0.7; do
            
            # Inner loops: GNN type, metric, loss
            for gnn in gcn gat sage; do
                for metric in acc qwk; do
                    for loss in ce dwce; do
                        
                        # Handle DWCE alpha values
                        if [ "$loss" = "dwce" ]; then
                            alpha_values=(0.1 0.5)
                        else
                            alpha_values=(0.0)  # Dummy value for CE (not used)
                        fi
                        
                        for alpha in "${alpha_values[@]}"; do
                            ((completed_experiments++))
                            
                            # Build experiment name
                            exp_name="${gnn}_h${hidden}_l${layers}_d${dropout}_${metric}_${loss}"
                            if [ "$loss" = "dwce" ]; then
                                exp_name="${exp_name}_a${alpha}"
                            fi
                            
                            echo "================================================" | tee -a ${LOG_FILE}
                            echo "Experiment ${completed_experiments}/${total_experiments}" | tee -a ${LOG_FILE}
                            echo "Config: ${exp_name}" | tee -a ${LOG_FILE}
                            echo "Time: $(date)" | tee -a ${LOG_FILE}
                            echo "================================================" | tee -a ${LOG_FILE}
                            
                            # Build command
                            cmd="python main_ordinal.py \
                                --dataset ${DATASET} \
                                --gnn ${gnn} \
                                --ordinal_loss ${loss} \
                                --metric ${metric} \
                                --lr ${LR} \
                                --epochs ${EPOCHS} \
                                --local_layers ${layers} \
                                --hidden_channels ${hidden} \
                                --dropout ${dropout} \
                                --runs ${RUNS} \
                                --device ${GPU} \
                                --res \
                                --bn"
                            
                            # Add alpha for DWCE
                            if [ "$loss" = "dwce" ]; then
                                cmd="${cmd} --dwce_alpha ${alpha}"
                            fi
                            
                            # Execute command
                            echo "Command: ${cmd}" | tee -a ${LOG_FILE}
                            echo "" | tee -a ${LOG_FILE}
                            
                            eval ${cmd} 2>&1 | tee -a ${LOG_FILE}
                            
                            exit_code=${PIPESTATUS[0]}
                            
                            if [ ${exit_code} -eq 0 ]; then
                                echo "✓ Experiment completed successfully" | tee -a ${LOG_FILE}
                            else
                                echo "✗ Experiment failed with exit code ${exit_code}" | tee -a ${LOG_FILE}
                            fi
                            
                            echo "" | tee -a ${LOG_FILE}
                            
                            # Small delay between experiments
                            # sleep 2
                        done
                    done
                done
            done
        done
    done
done

echo "========================================" | tee -a ${LOG_FILE}
echo "All experiments completed!" | tee -a ${LOG_FILE}
echo "Total: ${completed_experiments}/${total_experiments}" | tee -a ${LOG_FILE}
echo "End time: $(date)" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}