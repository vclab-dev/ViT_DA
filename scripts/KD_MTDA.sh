CUDA_VISIBLE_DEVICES=0 python3 KD_MTDA.py --arch rn50 --batch_size 64 --source Art --use_wandb 1 --epochs 100
CUDA_VISIBLE_DEVICES=0 python3 KD_MTDA.py --arch rn50 --batch_size 64 --source Clipart --use_wandb 1 --epochs 100
CUDA_VISIBLE_DEVICES=0 python3 KD_MTDA.py --arch rn50 --batch_size 64 --source Product --use_wandb 1 --epochs 100
CUDA_VISIBLE_DEVICES=0 python3 KD_MTDA.py --arch rn50 --batch_size 64 --source RealWorld --use_wandb 1 --epochs 100