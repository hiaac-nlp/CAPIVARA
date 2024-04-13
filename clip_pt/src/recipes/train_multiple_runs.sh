num_runs=3
base_checkpoint_path="/hadatasets/alef.ferreira/CAPIVARA/clip_pt/checkpoints"

# config_path="/hadatasets/alef.ferreira/CAPIVARA/clip_pt/experiment_setup/fine_tune_pracegover_filtered_capivara_opt.yaml"
config_path="/hadatasets/alef.ferreira/CAPIVARA/clip_pt/experiment_setup/fine_tune_pracegover_augmented_capivara_opt.yaml"
gpu=3

for i in $(seq 1 $num_runs)
do
    echo "======= RUN $i ========"
    python3 main_open_clip.py \
        --checkpoint-dir=$base_checkpoint_path \
        --config_path=$config_path \
        --gpu=$gpu
done