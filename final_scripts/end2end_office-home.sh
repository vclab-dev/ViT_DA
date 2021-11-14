python STDA_fbnm.py --gpu_id '1' --s 0 --t 1 2 3 --max_epoch 2 --interval 2 --batch_size 128 --dset office-home --net deit_s --wandb 0 --fbnm_par 4.0 --output_src src_train --output delete --suffix fbnm_rlccsoft_ep50 --rlcc 1 --soft_pl 1
python STDA_fbnm.py --gpu_id 1 --s 1 --t 0 2 --max_epoch 50 --interval 50 --batch_size 56 --dset office --net deit_s --wandb 1 --fbnm_par 4.0 --output_src src_train --output temp/deit/STDA_wt_fbnm_rlccsoft_ep50 --suffix fbnm_rlccsoft_ep50 --rlcc 1 --soft_pl 1


# Ablation
# python STDA_hp_cls_const_fbnm_with_grad.py --gpu_id 1 --s 0 --t 1 --max_epoch 50 --interval 50 --batch_size 56 --dset office-home --net deit_s --wandb 1 --fbnm 0 --gent 1 --ent 1 --fbnm_par 0.0 --grad_norm 1 --output_src src_train --output temp/deit/abla/STDA_wt_im_ent --suffix abla_im_ent
# python STDA_hp_cls_const_fbnm_with_grad.py --gpu_id 0 --s 0 --t 1 --max_epoch 50 --interval 50 --batch_size 56 --dset office-home --net deit_s --wandb 1 --fbnm 0 --gent 1 --ent 1 --fbnm_par 0.0 --grad_norm 1 --output_src src_train --output temp/deit/abla/STDA_wt_im_ent_rlcc_soft --suffix abla_im_ent_rlcc_soft --rlcc --soft_pl
