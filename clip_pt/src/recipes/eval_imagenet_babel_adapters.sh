# list=("wgkvuhsl" "chr7oxko" "p4shj7v6")
# list=("7lqtadvb" "mcsfsxvc" "7yyfdtno")
list=("wldvmmk0" "1xk4t0i8" "nijhruqa")

# exp_name="CC3M-Anotacoes_Originais-Adapters"
# exp_name="CC3M-10_Captions_Gerados-Adapters"
exp_name="CC3M-1500_steps-Adapters"

for i in ${list[@]}
do
    echo "======= $i ========"
    python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "DATASET_PATH" --save-dir "./" --languages "MANUAL-TRANSLATED-PT" --adapter "$i" --open-clip True --exp-name $exp_name
    python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "DATASET_PATH" --save-dir "./" --languages "PT" --adapter "$i" --open-clip True --exp-name $exp_name
    python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "DATASET_PATH" --save-dir "./" --languages "EN" --adapter "$i" --open-clip True --exp-name $exp_name
done