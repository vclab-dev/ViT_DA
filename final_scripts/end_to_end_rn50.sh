# Training end to end with Art to others using resnet 101 as teacher and resnet50 as student
python image_source_final.py --output san --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --net resnet101 &&
python STDA.py --cls_par 0.3 --batch_size 64 --dset office-home --gpu_id 0 --output delete --output_src san --s 0 --wandb 1 --net resnet101 --batch_size 128 &&
python KD_MTDA_all_dataset.py -s Art --dset office-home --txt ./data/office-home --save delete_2 -l ./delete/STDA/office-home --wandb 0 --batch_size 128 --arch_teacher resnet101