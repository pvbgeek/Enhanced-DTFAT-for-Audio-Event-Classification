#!/bin/bash
set -euo pipefail

# ================================================================
# Usage:
#   bash train_run.sh <variant> [seed]
#
#   variant: baseline | attn_pool | hybrid_stem | multiscale | all_combined
#            attn_pool_freeze | hybrid_stem_freeze | multiscale_freeze | all_combined_freeze
#            all | all_freeze
#   seed   : 0 | 1 | 2 | all (default: all)
#
# Examples:
#   bash train_run.sh baseline 0
#   bash train_run.sh attn_pool_freeze 0
#   bash train_run.sh all_freeze
# ================================================================

VARIANT=${1:-"all"}
SEED_ARG=${2:-"all"}

valid_variants=(
    "baseline" "attn_pool" "hybrid_stem" "multiscale" "all_combined"
    "attn_pool_freeze" "hybrid_stem_freeze" "multiscale_freeze" "all_combined_freeze"
    "all" "all_freeze"
)
valid=0
for v in "${valid_variants[@]}"; do
    if [[ "$VARIANT" == "$v" ]]; then valid=1; break; fi
done
if [[ $valid -eq 0 ]]; then
    echo "ERROR: Unknown variant '${VARIANT}'"
    echo "Valid: baseline | attn_pool | hybrid_stem | multiscale | all_combined"
    echo "       attn_pool_freeze | hybrid_stem_freeze | multiscale_freeze | all_combined_freeze"
    echo "       all | all_freeze"
    exit 1
fi

if [[ "$SEED_ARG" != "all" && "$SEED_ARG" != "0" && "$SEED_ARG" != "1" && "$SEED_ARG" != "2" ]]; then
    echo "ERROR: Unknown seed '${SEED_ARG}'. Valid: 0 | 1 | 2 | all"
    exit 1
fi

echo "========================================"
echo "  variant : ${VARIANT}"
echo "  seed    : ${SEED_ARG}"
echo "========================================"

# ---- Config ----
dataset=audioset
imagenetpretrain=True
lr=2e-4
epoch=40
batch_size=32
n_workers=8

python_bin="$(which python)"

working_dir=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/egs/audioset
label_csv=${working_dir}/data/class_labels_indices.csv

tr_data=/WAVE/projects/oignat_lab/Parth-Personal/Dataset/balanced_train_converted.json
te_data=/WAVE/projects/oignat_lab/Parth-Personal/Dataset/eval_converted_clean.json

run_py=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/src/run.py
eval_py=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/src/eval.py
plot_py=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/src/plot_results.py

base_exp_dir=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/experiments

# ---- Step 0: rebuild eval manifest ----
te_data_fixed=/WAVE/projects/oignat_lab/Parth-Personal/Dataset/eval_converted_clean_exists.json
${python_bin} - <<'PY'
import json, os
src = "/WAVE/projects/oignat_lab/Parth-Personal/Dataset/eval_converted_clean.json"
dst = "/WAVE/projects/oignat_lab/Parth-Personal/Dataset/eval_converted_clean_exists.json"
d = json.load(open(src))
before = len(d.get("data", []))
d["data"] = [x for x in d.get("data", []) if os.path.exists(x.get("wav",""))]
after = len(d["data"])
json.dump(d, open(dst,"w"), indent=2)
print(f"[manifest] {before} -> {after} | wrote {dst}")
PY

# ================================================================
# Helper: run one experiment
# Args: variant, reverse_aug, freeze, seed
# ================================================================
run_experiment() {
    local variant=$1
    local reverse_aug=$2
    local freeze=$3
    local seed=$4
    local exp_dir=${base_exp_dir}/${variant}_seed${seed}

    # Strip _freeze suffix to get the actual model_variant
    local model_variant="${variant/_freeze/}"

    mkdir -p "${exp_dir}"

    if [[ -f "${exp_dir}/models/best_audio_model.pth" ]]; then
        echo "===== ${variant} seed ${seed} already has best model. Skipping training. ====="
    else
        echo "===== Training variant=${variant} freeze=${freeze} seed=${seed} | exp_dir=${exp_dir} ====="
        rm -rf "${exp_dir}"
        mkdir -p "${exp_dir}"

        ${python_bin} ${run_py} \
            --dataset ${dataset} \
            --data-train ${tr_data} \
            --data-val ${te_data_fixed} \
            --label-csv ${label_csv} \
            --exp-dir ${exp_dir} \
            --lr ${lr} \
            --n-epochs ${epoch} \
            --batch-size ${batch_size} \
            --num-workers ${n_workers} \
            --freqm 48 \
            --timem 192 \
            --mixup 0.5 \
            --imagenet_pretrain ${imagenetpretrain} \
            --metrics mAP \
            --loss BCE \
            --save_model True \
            --seed ${seed} \
            --model_variant ${model_variant} \
            --reverse_aug ${reverse_aug} \
            --freeze ${freeze} \
            2>&1 | tee "${exp_dir}/train.log"
    fi

    # Eval on best model
    if [[ -f "${exp_dir}/models/best_audio_model.pth" ]]; then
        echo "===== Eval-only variant=${variant} seed=${seed} ====="
        ${python_bin} -u ${eval_py} \
            --dataset ${dataset} \
            --data-train not_needed \
            --data-val ${te_data_fixed} \
            --working_dir ${working_dir} \
            --label-csv ${label_csv} \
            --lr 5e-4 \
            --n-epochs 1 \
            --batch-size 64 \
            --num-workers ${n_workers} \
            --save_model False \
            --imagenet_pretrain ${imagenetpretrain} \
            --model_variant ${model_variant} \
            --as2m_ckpt "${exp_dir}/models/best_audio_model.pth" \
            2>&1 | tee "${exp_dir}/eval_best.log"
    fi

    echo "===== Plotting curves for variant=${variant} seed=${seed} ====="
    ${python_bin} ${plot_py} --exp-dir "${exp_dir}" 2>&1 | tee "${exp_dir}/plot.log"
}

# ================================================================
# Helper: run all seeds for a variant
# ================================================================
run_variant() {
    local variant=$1
    local freeze=$2
    echo "########## Running: ${variant} (freeze=${freeze}) ##########"
    if [[ "$SEED_ARG" == "all" ]]; then
        for seed in 0 1 2; do
            run_experiment "${variant}" True ${freeze} ${seed}
        done
    else
        run_experiment "${variant}" True ${freeze} ${SEED_ARG}
    fi
}

# ================================================================
# Dispatch
# ================================================================
case "$VARIANT" in
    "all")
        run_variant "baseline"       False
        run_variant "attn_pool"      False
        run_variant "hybrid_stem"    False
        run_variant "multiscale"     False
        run_variant "all_combined"   False
        ;;
    "all_freeze")
        run_variant "attn_pool_freeze"    True
        run_variant "hybrid_stem_freeze"  True
        run_variant "multiscale_freeze"   True
        run_variant "all_combined_freeze" True
        ;;
    *_freeze)
        run_variant "${VARIANT}" True
        ;;
    *)
        run_variant "${VARIANT}" False
        ;;
esac

echo "========================================"
echo "DONE: variant=${VARIANT} seed=${SEED_ARG}"
echo "========================================"