list=("57mwx8py" "ktuk89di" "swzhiavx" "0frz4eth" "kvmigydr" "p8srqfk4" "mlwf4vdn")

for i in ${list[@]}
do
   echo "======= $i ========"
   python3 evaluate/zero_shot_retrieval_webdatasets.py --model-path "./CLIP-PT/$i/checkpoints/last.ckpt" --dataset-path "/hadatasets/clip_pt/final_webdatasets/flickr30k_val/00000.tar" --batch 100 --gpu 1 --translation "google" --distill False --open-clip True
done

