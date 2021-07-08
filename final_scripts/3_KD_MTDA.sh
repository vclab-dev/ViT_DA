# DomainNet
## 
CUDA_VISIBLE_DEVICES=0 python KD_MTDA_all_dataset.py -s sketch --dset domain_net -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net

# Office-Home

CUDA_VISIBLE_DEVICES=0 python KD_MTDA_all_dataset.py -s Clipart --dset office-home -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net

# Office-31

CUDA_VISIBLE_DEVICES=0 python KD_MTDA_all_dataset.py -s amazon --dset office -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net

# PACS
CUDA_VISIBLE_DEVICES=0 python KD_MTDA_all_dataset.py -s art_painting --dset pacs -t ./data/domain_net --save delete -l ./BMVC_STDA_DomainNet/uda/domain_net
