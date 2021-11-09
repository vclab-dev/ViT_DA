# python image_source_final.py --gpu_id 0 --s 0 --max_epoch 50 --interval 25 --dset office-home --net resnet50 --wandb 1 --output rn50/src_train
# python image_source_final.py --gpu_id 0 --s 1 --max_epoch 50 --interval 25 --dset office-home --net resnet50 --wandb 1 --output rn50/src_train
# python image_source_final.py --gpu_id 0 --s 2 --max_epoch 50 --interval 25 --dset office-home --net resnet50 --wandb 1 --output rn50/src_train
python image_source_final.py --gpu_id 2 --s 3 --max_epoch 50 --interval 25 --dset office-home --net resnet50 --wandb 1 --output rn50/src_train


# python STDA_hp_cls_const_fbnm_with_grad.py --gpu_id 1 --s 0 --max_epoch 30 --interval 30 --batch_size 64 --dset office-home --net resnet50 --wandb 1 --fbnm_par 4.0 --grad_norm 1 --output_src rn50/src_train --output rn50/STDA_wt_fbnm_with_grad_rlcc_soft_with_stg --suffix fbnm_grad_rlcc_soft_stg_rn50 --rlcc --soft_pl
# python STDA_hp_cls_const_fbnm_with_grad.py --gpu_id 1 --s 1 --max_epoch 30 --interval 30 --batch_size 64 --dset office-home --net resnet50 --wandb 1 --fbnm_par 4.0 --grad_norm 1 --output_src rn50/src_train --output rn50/STDA_wt_fbnm_with_grad_rlcc_soft_with_stg --suffix fbnm_grad_rlcc_soft_stg_rn50 --rlcc --soft_pl
# python STDA_hp_cls_const_fbnm_with_grad.py --gpu_id 1 --s 2 --max_epoch 30 --interval 30 --batch_size 64 --dset office-home --net resnet50 --wandb 1 --fbnm_par 4.0 --grad_norm 1 --output_src rn50/src_train --output rn50/STDA_wt_fbnm_with_grad_rlcc_soft_with_stg --suffix fbnm_grad_rlcc_soft_stg_rn50 --rlcc --soft_pl
python STDA_hp_cls_const_fbnm_with_grad.py --gpu_id 2 --s 3 --max_epoch 30 --interval 30 --batch_size 64 --dset office-home --net resnet50 --wandb 1 --fbnm_par 4.0 --grad_norm 1 --output_src rn50/src_train --output rn50/STDA_wt_fbnm_with_grad_rlcc_soft_with_stg --suffix fbnm_grad_rlcc_soft_stg_rn50 --rlcc --soft_pl
