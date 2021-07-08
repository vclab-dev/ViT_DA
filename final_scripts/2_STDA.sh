
################ STDA for sketch (DomiainNet) ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch', 'real']

python STDA.py --cls_par 0.3 --batch_size 64 --dset domain_net --gpu_id 0 --output delete --output_src san --s 4 --t 0 --wandb 0 --worker 0 

python STDA.py --cls_par 0.3 --batch_size 64 --dset domain_net --gpu_id 0 --output delete --output_src san --s 4 --t 1 --wandb 0 --worker 0

python STDA.py --cls_par 0.3 --batch_size 64 --dset domain_net --gpu_id 0 --output delete --output_src san --s 4 --t 2 --wandb 0 --worker 0

python STDA.py --cls_par 0.3 --batch_size 64 --dset domain_net --gpu_id 0 --output delete --output_src san --s 4 --t 3 --wandb 0 --worker 0

python STDA.py --cls_par 0.3 --batch_size 64 --dset domain_net --gpu_id 0 --output delete --output_src san --s 4 --t 5 --wandb 0 --worker 0 

################ STDA for webcam (Office31)  ['amazon', 'dslr', 'webcam']

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office --gpu_id 0 --output delete --output_src san --s 2 --t 0 --wandb 0 --worker 4 

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office --gpu_id 0 --output delete --output_src san --s 2 --t 1 --wandb 0 --worker 4

################ STDA for Art (OfficeHome) ['Art', 'Clipart', 'Product', 'RealWorld']

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office-home --gpu_id 0 --output delete --output_src san --s 0 --t 1 --wandb 0 --worker 0

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office-home --gpu_id 0 --output delete --output_src san --s 0 --t 2 --wandb 0 --worker 0

# python STDA.py --cls_par 0.3 --batch_size 64 --dset office-home --gpu_id 0 --output delete --output_src san --s 0 --t 3 --wandb 0 --worker 0

################ STDA for art_painting (PACS) ['art_painting', 'cartoon', 'photo', 'sketch']

# python STDA.py --cls_par 0.3 --batch_size 64 --dset pacs --gpu_id 0 --output delete --output_src san --s 0 --t 1 --wandb 0 --worker 0

# python STDA.py --cls_par 0.3 --batch_size 64 --dset pacs --gpu_id 0 --output delete --output_src san --s 0 --t 2 --wandb 0 --worker 0

# python STDA.py --cls_par 0.3 --batch_size 64 --dset pacs --gpu_id 0 --output delete --output_src san --s 0 --t 3 --wandb 0 --worker 0
