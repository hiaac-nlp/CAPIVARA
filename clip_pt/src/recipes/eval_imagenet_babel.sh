# list=("8bvxl7y9" "nsnr05s6" "3n0ds4f0")
# list=("2kmj76gi" "upxpipp8" "mp5acwpw")
# list=("rejut8sg" "l7hky6yf" "awdqigwi")
# list=("71drziaz" "4uq6a4mp" "kyr94t7j")
# list=("sook0bcf" "95mmsj9q" "7aacn3tc")
list=("wyza08z5" "06jftu81" "rgaj0rsb")

# exp_name="CC3M-Anotacoes_Originais"
# exp_name="CC3M-10_Captions_Gerados"
# exp_name="CC3M_Filtrado-Anotacoes_Originais"
# exp_name="CC3M_Filtrado-10_Captions_Gerados"
# exp_name="CC3M-1_Caption_Gerado"
exp_name="CC3M-5_Captions_Gerados"

for i in ${list[@]}
do
    echo "======= $i ========"
    python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "MANUAL-TRANSLATED-PT" --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt" --open-clip True --exp-name $exp_name
    python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "PT" --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt" --open-clip True --exp-name $exp_name
    python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "EN" --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt" --open-clip True --exp-name $exp_name
done