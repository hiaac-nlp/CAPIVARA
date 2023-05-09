list=("32fxdgp0" "jflhnrv1")

for i in ${list[@]}
do
   echo "======= $i ========"
   python evaluate/zero_shot_retrieval_webdatasets.py --model-path "./CLIP-PT/$i/checkpoints/last.ckpt" --dataset-path "/hadatasets/clip_pt/final_webdatasets/flickr30k_val/00000.tar" --batch 100 --gpu $1 --translation "google" --distill True
done

