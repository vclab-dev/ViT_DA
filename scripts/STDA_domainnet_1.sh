################ STDA  ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch', 'real']

# python STDA_hp.py --batch_size 80 --dset domain_net --gpu_id 0 --output optimised_STDA_wt --max_epoch 50 --interval 50 --output_src san --s 4 --wandb 1 --worker 4 --suffix rlcc_soft --soft_pl --earlystop 1
# python image_source_final.py --gpu_id 0 --s 0 --max_epoch 30 --interval 15 --batch_size 256 --dset domain_net --net deit_b --wandb 1 --output src_train
# python image_source_final.py --gpu_id 0 --s 1 --max_epoch 30 --interval 15 --batch_size 256 --dset domain_net --net deit_b --wandb 1 --output src_train
# python image_source_final.py --gpu_id 0 --s 2 --max_epoch 30 --interval 15 --batch_size 256 --dset domain_net --net deit_b --wandb 1 --output src_train
# python image_source_final.py --gpu_id 0 --s 3 --max_epoch 30 --interval 15 --batch_size 256 --dset domain_net --net deit_b --wandb 1 --output src_train
# python image_source_final.py --gpu_id 0 --s 4 --max_epoch 30 --interval 15 --batch_size 256 --dset domain_net --net deit_b --wandb 1 --output src_train
# python image_source_final.py --gpu_id 0 --s 5 --max_epoch 30 --interval 15 --batch_size 256 --dset domain_net --net deit_b --wandb 1 --output src_train

python3 STDA_fbnm.py --gpu_id '0,1,2' --s 3 --t 0 1 2 4 5 --max_epoch 5 --interval 5 --batch_size 48 --dset domain_net --net vit --wandb 1 --fbnm_par 4.0 --output_src src_train --output weights/STDA_wt_fbnm_rlccsoft --suffix fbnm_rlcc_soft --rlcc 1 --soft_pl 1
