# python image_source_final.py --output san --gpu_id 0 --dset office-home --max_epoch 50 --s 3 --net deit_s
# python image_source_final.py --output san --gpu_id 0 --dset office-home --max_epoch 50 --s 1 --net deit_s
# python image_source_final.py --output san --gpu_id 0 --dset office-home --max_epoch 50 --s 2 --net deit_s

python STDA_hp.py --batch_size 64 --dset office-home --net deit_s --gpu_id 0 --output optimised_STDA_wt --max_epoch 50 --interval 50 --output_src san --s 0 --wandb 1 --worker 4 -suffix 65x65 v2
# python STDA_hp.py --batch_size 64 --dset office-home --net deit_s --gpu_id 0 --output optimised_STDA_wt_soft_pl --max_epoch 1 --interval 5 --output_src san --s 0 --wandb 0 --worker 4 --soft_pl # > debug.txt
