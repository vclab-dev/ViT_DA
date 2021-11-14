python3 ocda.py --gpu_id '2,0,1' --dset office-home --net deit_s --s 0 --t 2 --batch_size 384 --epoch 25 --interval 5 --suffix ocda --save_weights ocda_wts --wandb 1 --txt_folder test_target/no_grad
python3 ocda.py --gpu_id '2,0,1' --dset office-home --net deit_s --s 0 --t 3 --batch_size 384 --epoch 25 --interval 5 --suffix ocda --save_weights ocda_wts --wandb 1 --txt_folder test_target/no_grad
