list=("3cavs4uj" "2pqi2s2g" "ch2u16t2" "n702m8ga" "308t9rhg" "pk1sesbc" "sr8onoru" "o2gb5ls9" "hy25ol3h" "sc7036en" "bntaqx4c" "06wezfw3" "7kbzw1of" "uxoaa9f0" "20cg4c7i" "tk3krumn" "e8m7d2hh" "iusuqm3k" "rvytjuve" "qi263rbf" "xod21u43" "fg43cqdt" "6ox8n3x1" "witutsv6" "vfkr5awt" "q7y8d52o" "p0pgzqpb" "s30xkg3h" "kogle15q" "cmo00utd" "pxvry8l1" "o6r6jb3d" "q05dlir3" "8lmb67e1" "hrzl7oi6" "9iicx2rs" "pvzexdzv" "3d9c3imp" "e6fk39tx" "ur6dvbvh")

for i in ${list[@]}
do
   echo "======= $i ========"
   python ../evaluate/zero_shot_retrieval_webdatasets.py --model-path "./CLIP-PT/$i/checkpoints/last.ckpt" --dataset-path "/hadatasets/clip_pt/final_webdatasets/pracegover_val/{00000..00007}.tar" --batch 100 --gpu $1
done


