# CUDA_VISIBLE_DEVICES=0 python3 MixUp_KD.py --dset office-home --s 0 --batch_size 64 --epoch 100 --suffix GradNormCSV --wandb 0

# CUDA_VISIBLE_DEVICES=1 python3 MixUp_KD.py --dset office-home --s 0 --batch_size 80 --epoch 100 --interval 10 --suffix noGrad --save_weights MTDA_weights --wandb 1 --txt_folder test_target/no_grad
CUDA_VISIBLE_DEVICES=1 python3 MixUp_KD.py --dset office-home --s 1 --batch_size 80 --epoch 50 --interval 10 --suffix noGrad --save_weights MTDA_weights --wandb 1 --txt_folder test_target/no_grad