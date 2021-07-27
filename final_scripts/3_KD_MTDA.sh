# DomainNet

python KD_MTDA_all_dataset.py --gpu_id 0 -s sketch --dset domain_net -t ./data/domain_net --save optimised_rn101_MTDA_wt --arch rn101 -l optimised_STDA_wt/STDA/domain_net --wandb 1 

python KD_MTDA_all_dataset.py --gpu_id 0 -s infograph --dset domain_net -t ./data/domain_net --save optimised_rn101_MTDA_wt --arch rn101 -l optimised_STDA_wt/STDA/domain_net --wandb 1 --batch_size 128

python KD_MTDA_all_dataset.py --gpu_id 0 -s quickdraw --dset domain_net -t ./data/domain_net --save optimised_rn101_MTDA_wt --arch rn101 -l optimised_STDA_wt/STDA/domain_net --wandb 1 --batch_size 128

python KD_MTDA_all_dataset.py --gpu_id 0 -s clipart --dset domain_net -t ./data/domain_net --save optimised_rn101_MTDA_wt --arch rn101 -l optimised_STDA_wt/STDA/domain_net --wandb 1 

python KD_MTDA_all_dataset.py --gpu_id 0 -s painting --dset domain_net -t ./data/domain_net --save optimised_rn101_MTDA_wt --arch rn101 -l optimised_STDA_wt/STDA/domain_net --wandb 1 

python KD_MTDA_all_dataset.py --gpu_id 0 -s real --dset domain_net -t ./data/domain_net --save optimised_rn101_MTDA_wt --arch rn101 -l optimised_STDA_wt/STDA/domain_net --wandb 1 



# Office-Home

python KD_MTDA_all_dataset.py -s Clipart --dset office-home -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net --wandb 0

# Office-31

python KD_MTDA_all_dataset.py -s amazon --dset office -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net

# PACS
python KD_MTDA_all_dataset.py -s art_painting --dset pacs -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net
