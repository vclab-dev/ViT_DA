# CUDA_VISIBLE_DEVICES=0 python3 MixUp_KD.py --dset office-home --s 0 --batch_size 64 --epoch 100 --suffix GradNormCSV --wandb 0
CUDA_VISIBLE_DEVICES=0 python3 MixUp_KD.py --dset office-home --s 1 --batch_size 64 --epoch 100 --suffix GradNormCSV --wandb 1
CUDA_VISIBLE_DEVICES=1 python3 MixUp_KD.py --dset office-home --s 2 --batch_size 64 --epoch 100 --suffix GradNormCSV --wandb 1
CUDA_VISIBLE_DEVICES=0 python3 MixUp_KD.py --dset office-home --s 3 --batch_size 64 --epoch 100 --suffix GradNormCSV --wandb 1