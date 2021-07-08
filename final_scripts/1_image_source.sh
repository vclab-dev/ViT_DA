#Office-31
python image_source_final.py --trte val --output ckps/source/  --gpu_id 0 --dset office --max_epoch 100 --s 0 --net vit --se True ;

#office-home
python image_source_final.py --trte val --output ckps/source/  --gpu_id 0 --dset office-home --max_epoch 100 --s 0 --net vit --se True ;

#pacs
python image_source_final.py --trte val --output ckps/source/  --gpu_id 0 --dset pacs --max_epoch 100 --s 0 --net vit --se True ;

#domain_net
python image_source_final.py --trte val --output ckps/source/  --gpu_id 0 --dset domain-net --max_epoch 100 --s 0 --net vit --se True ;