exp_name="CAPIVARA-CLIP-PT-IMAGENET-BABEL"

python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "MANUAL-TRANSLATED-PT" --open-clip True --exp-name $exp_name
python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "PT" --open-clip True --exp-name $exp_name
python3 ../evaluate/zero_shot_imagenet_babel.py --gpu 5 --imagenet_folder "/hadatasets/clip_pt/images/imagenet" --save-dir "./" --languages "EN" --open-clip True --exp-name $exp_name