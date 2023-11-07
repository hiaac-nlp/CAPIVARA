gpu=4
exp_name="CAPIVARA-CLIP-PT-IMAGENET-BABEL-ADAPTERS"

python3 ../evaluate/zero_shot_imagenet_babel.py --gpu $gpu --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "MANUAL-TRANSLATED-PT" --adapter "hiaac-nlp/CAPIVARA-LoRA" --open-clip True --exp-name $exp_name
python3 ../evaluate/zero_shot_imagenet_babel.py --gpu $gpu --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "PT" --adapter "hiaac-nlp/CAPIVARA-LoRA" --open-clip True --exp-name $exp_name
python3 ../evaluate/zero_shot_imagenet_babel.py --gpu $gpu --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "EN" --adapter "hiaac-nlp/CAPIVARA-LoRA" --open-clip True --exp-name $exp_name