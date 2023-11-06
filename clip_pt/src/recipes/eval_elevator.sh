gpu=4
exp_name="CAPIVARA-CLIP-PT-ELEVATER"

export LANGUAGE="pt-BR"
# export LANGUAGE="en"
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "cifar-10" --datapath "../../../../../ELEVATER/all_zip/cifar-10/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "gtsrb" --datapath "../../../../../ELEVATER/all_zip/gtsrb/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "caltech-101" --datapath "../../../../../ELEVATER/all_zip/caltech-101/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "cifar-100" --datapath "../../../../../ELEVATER/all_zip/cifar-100/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "country211" --datapath "../../../../../ELEVATER/all_zip/country211/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "fgvc-aircraft-2013b-variants102" --datapath "../../../../../ELEVATER/all_zip/fgvc-aircraft-2013b-variants102/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "oxford-flower-102" --datapath "../../../../../ELEVATER/all_zip/oxford-flower-102/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "food-101" --datapath "../../../../../ELEVATER/all_zip/food-101/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "kitti-distance" --datapath "../../../../../ELEVATER/all_zip/kitti-distance/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "mnist" --datapath "../../../../../ELEVATER/all_zip/mnist/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "patch-camelyon" --datapath "../../../../../ELEVATER/all_zip/patch-camelyon/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "resisc45_clip" --datapath "../../../../../ELEVATER/all_zip/resisc45_clip/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "stanford-cars" --datapath "../../../../../ELEVATER/all_zip/stanford-cars/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "voc-2007-classification" --datapath "../../../../../ELEVATER/all_zip/voc-2007-classification/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "oxford-iiit-pets" --datapath "../../../../../ELEVATER/all_zip/oxford-iiit-pets/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "eurosat_clip" --datapath "../../../../../ELEVATER/all_zip/eurosat_clip/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "hateful-memes" --datapath "../../../../../ELEVATER/all_zip/hateful-memes/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "rendered-sst2" --datapath "../../../../../ELEVATER/all_zip/rendered-sst2/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "dtd" --datapath "../../../../../ELEVATER/all_zip/dtd/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name
python3 ../evaluate/zero_shot_elevater.py --gpu $gpu --open-clip True --dataset "fer-2013" --datapath "../../../../../ELEVATER/all_zip/fer-2013/test" --save-dir "../evaluate/elevater/output" --exp-name $exp_name

