# list=("8bvxl7y9" "nsnr05s6" "3n0ds4f0")
# list=("2kmj76gi" "upxpipp8" "mp5acwpw")
# list=("rejut8sg" "l7hky6yf" "awdqigwi")
# list=("71drziaz" "4uq6a4mp" "kyr94t7j")
# list=("sook0bcf" "95mmsj9q" "7aacn3tc")
list=("wyza08z5" "06jftu81" "rgaj0rsb")

gpu=5
# exp_name="CC3M_Filtrado-10_Captions_Gerados"
# exp_name="CC3M-1_Caption_Gerado"
exp_name="CC3M-5_Captions_Gerados"

export LANGUAGE="pt-BR"

for i in ${list[@]}
do
    echo "======= $i ========"
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "cifar-10" --datapath "../../../all_zip/cifar-10/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "gtsrb" --datapath "../../../all_zip/gtsrb/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "caltech-101" --datapath "../../../all_zip/caltech-101/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "cifar-100" --datapath "../../../all_zip/cifar-100/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "country211" --datapath "../../../all_zip/country211/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "fgvc-aircraft-2013b-variants102" --datapath "../../../all_zip/fgvc-aircraft-2013b-variants102/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "oxford-flower-102" --datapath "../../../all_zip/oxford-flower-102/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "food-101" --datapath "../../../all_zip/food-101/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "kitti-distance" --datapath "../../../all_zip/kitti-distance/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "mnist" --datapath "../../../all_zip/mnist/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "patch-camelyon" --datapath "../../../all_zip/patch-camelyon/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "resisc45_clip" --datapath "../../../all_zip/resisc45_clip/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "stanford-cars" --datapath "../../../all_zip/stanford-cars/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "voc-2007-classification" --datapath "../../../all_zip/voc-2007-classification/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "oxford-iiit-pets" --datapath "../../../all_zip/oxford-iiit-pets/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "eurosat_clip" --datapath "../../../all_zip/eurosat_clip/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "hateful-memes" --datapath "../../../all_zip/hateful-memes/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "rendered-sst2" --datapath "../../../all_zip/rendered-sst2/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "dtd" --datapath "../../../all_zip/dtd/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 evaluate/zero_shot_elevater.py --model-path "/hahomes/gabriel.santos/CLIP-PtBr/clip_pt/src/CLIP-PT/$i/checkpoints/last.ckpt"  --gpu $gpu --open-clip True --dataset "fer-2013" --datapath "../../../all_zip/fer-2013/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
done

