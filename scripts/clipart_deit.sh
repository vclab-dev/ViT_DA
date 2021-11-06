source=1
python image_source_final.py --output san --gpu_id 1 --dset office-home --max_epoch 50 --s $source --net deit_s &&
python STDA.py --cls_par 0.3 --batch_size 64 --dset office-home --gpu_id 1 --output delete --output_src san --s $source --wandb 1 --net deit_s --batch_size 32 --max_epoch 20 &&
python KD_MTDA_all_dataset.py -s Clipart --dset office-home --txt ./data/office-home --save delete_2 -l ./delete/STDA/office-home --wandb 1 --arch_teacher deit_s --gpu_id 1
