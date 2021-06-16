# CUDA_VISIBLE_DEVICES=1 python3 KD_MTDA.py --batch_size 24 --source Clipart --use_wandb 1 --epochs 100
CUDA_VISIBLE_DEVICES=0 python3 KD_MTDA.py --arch rn50 --batch_size 24 --source Clipart --use_wandb 1 --epochs 100
