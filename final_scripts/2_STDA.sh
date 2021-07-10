
################ STDA for sketch (DomiainNet) ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch', 'real']

python STDA.py --batch_size 16 --dset domain_net --gpu_id 0 --output optimised_STDA_wt --max_epoch 1 --output_src san --s 5 --wandb 0 --worker 4 

################ STDA for webcam (Office31)  ['amazon', 'dslr', 'webcam']

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office --gpu_id 0 --output delete --output_src san --s 2 --wandb 0 --worker 4 

################ STDA for Art (OfficeHome) ['Art', 'Clipart', 'Product', 'RealWorld']

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office-home --gpu_id 0 --output delete --output_src san --s 0 --wandb 0 --worker 0

################ STDA for art_painting (PACS) ['art_painting', 'cartoon', 'photo', 'sketch']

# python STDA.py --cls_par 0.3 --batch_size 64 --dset pacs --gpu_id 0 --output delete --output_src san --s 0 --wandb 0 --worker 0
