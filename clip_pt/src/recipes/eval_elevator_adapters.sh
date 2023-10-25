# list=("wgkvuhsl" "chr7oxko" "p4shj7v6")
# list=("7lqtadvb" "mcsfsxvc" "7yyfdtno")
list=("wldvmmk0" "1xk4t0i8" "nijhruqa")

# exp_name="CC3M-Anotacoes_Originais-Adapters"
# exp_name="CC3M-10_Captions_Gerados-Adapters"
exp_name="CC3M-1500_steps-Adapters"

gpu=5
export LANGUAGE="pt-BR"

for i in ${list[@]}
do
    echo "======= $i ========"
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "cifar-10" --datapath "../../../all_zip/cifar-10/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "gtsrb" --datapath "../../../all_zip/gtsrb/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "caltech-101" --datapath "../../../all_zip/caltech-101/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "cifar-100" --datapath "../../../all_zip/cifar-100/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "country211" --datapath "../../../all_zip/country211/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "fgvc-aircraft-2013b-variants102" --datapath "../../../all_zip/fgvc-aircraft-2013b-variants102/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "oxford-flower-102" --datapath "../../../all_zip/oxford-flower-102/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "food-101" --datapath "../../../all_zip/food-101/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "kitti-distance" --datapath "../../../all_zip/kitti-distance/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "mnist" --datapath "../../../all_zip/mnist/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "patch-camelyon" --datapath "../../../all_zip/patch-camelyon/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "resisc45_clip" --datapath "../../../all_zip/resisc45_clip/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "stanford-cars" --datapath "../../../all_zip/stanford-cars/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "voc-2007-classification" --datapath "../../../all_zip/voc-2007-classification/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "oxford-iiit-pets" --datapath "../../../all_zip/oxford-iiit-pets/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "eurosat_clip" --datapath "../../../all_zip/eurosat_clip/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "hateful-memes" --datapath "../../../all_zip/hateful-memes/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "rendered-sst2" --datapath "../../../all_zip/rendered-sst2/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "dtd" --datapath "../../../all_zip/dtd/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
    python3 ../evaluate/zero_shot_elevater.py --model-path "$i"  --gpu $gpu --open-clip True --adapter True --dataset "fer-2013" --datapath "../../../all_zip/fer-2013/test" --save-dir "evaluate/elevater/output" --exp-name $exp_name
done

