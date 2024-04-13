list=("t76rc49x" "gwjdh6r3" "9dzcf8tg" "ct6i9v4z" "bpzra9nv" "2mo69oy5")
base_checkpoint_path="/hadatasets/alef.ferreira/CAPIVARA/clip_pt/checkpoints/adapter_PEFT_checkpoints"
gpu=3

for i in ${list[@]}
do
    echo "======= $i ========"
    python3 ../evaluate/zero_shot_retrieval.py \
        --dataset-path "/hadatasets/clip_pt/final_webdatasets/pracegover_val/{00000..00007}.tar" \
        --adapter "$base_checkpoint_path/$i" \
        --batch 100 \
        --gpu $gpu \
        --open-clip True \
        --translation "google"
done