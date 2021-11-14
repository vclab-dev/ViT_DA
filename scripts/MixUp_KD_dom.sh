# python3 Simple_KD_DomainNet.py --gpu_id 0 --dset domain_net --net resnet101 --s 0 --batch_size 320 --epoch 9 --interval 3 --suffix simple_domainnet --save_weights MTDA_weights/resnet101 --wandb 1 --txt_folder test_target/no_grad
# CUDA_VISIBLE_DEVICES=1 python3 test_MixUp_KD_DomainNet.py --dset domain_net --net vit --s 5 --batch_size 512 --epoch 1 --interval 1 --suffix domainnet_noGrad --output MTDA_weights --wandb 0 --txt_folder test_target/no_grad

'''
python3 Simple_KD_DomainNet.py --gpu_id '2,0,1' --dset domain_net --net resnet101 --s 2 --batch_size 224 --epoch 9 --interval 3 --suffix simple_domainnet --save_weights MTDA_weights/resnet101 --wandb 1 --txt_folder test_target/no_grad
'''