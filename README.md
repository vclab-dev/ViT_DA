# Official implementation for TransDA
Official pytorch implementation for **Knowledge Distillation based Source-Free Multi-Target Domain Adaptation**.

## Prepare pretrain model
We choose R50-ViT-B_16 as our encoder.
```bash root transformerdepth
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz 
mkdir ./model/vit_checkpoint/imagenet21k 
mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## Dataset:
- Please manually download the datasets [Office](https://www.dropbox.com/sh/vja4cdimm0k2um3/AACCKNKV8-HVbEZDPDCyAyf_a?dl=0), [Office-Home](https://www.dropbox.com/sh/vja4cdimm0k2um3/AACCKNKV8-HVbEZDPDCyAyf_a?dl=0), PACS, DomainNet from the official websites, and modify the path of images in each '.txt' under the folder './data/'.
- For downloading DomainNet run `sh final_scripts/download_domain_net.sh`. Manually extract zip and keep directory structure as mentioned in [Dataset directory](#Dataset-directory)

## Training
### Stage 1: Source only Training

```sh
# Change parameters for different dataset
sh final_scripts/1_image_source.sh
```

### Stage 2: STDA training
```sh
# Change parameters for different dataset
# Manually set each STDA source and target
sh final_scripts/2_STDA.sh
```

### Stage 3: KD MTDA training
 ```sh
# Change parameters for different dataset
# Manually set each source
sh final_scripts/3_KD_MTDA.sh
 ```

## Prerequisites:
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, PIL, argparse, tqdm


## Dataset directory

```
├── domain_net
│   ├── clipart
│   ├── clipart.txt
│   ├── infograph
│   ├── infograph.txt
│   ├── painting
│   ├── painting.txt
│   ├── quickdraw
│   ├── quickdraw.txt
│   ├── real
│   ├── real.txt
│   ├── sketch
│   └── sketch.txt
├── office
│   ├── amazon
│   ├── amazon.txt
│   ├── dslr
│   ├── dslr.txt
│   ├── webcam
│   └── webcam.txt
├── office-home
│   ├── Art
│   ├── Art.txt
│   ├── Clipart
│   ├── Clipart.txt
│   ├── Product
│   ├── Product.txt
│   ├── Real_World
│   └── RealWorld.txt
├── office_home_mixed
│   ├── Art_Clipart_Product
│   ├── Art_Clipart_Product.txt
│   ├── Art_Clipart_Real_World
│   ├── Art_Clipart_Real_World.txt
│   ├── Art_Product_Real_World
│   ├── Art_Product_Real_World.txt
│   ├── Clipart_Product_Real_World
│   └── Clipart_Product_Real_World.txt
└── pacs
    ├── art_painting
    ├── art_painting.txt
    ├── cartoon
    ├── cartoon.txt
    ├── __MACOSX
    ├── photo
    ├── photo.txt
    ├── sketch
    └── sketch.txt
```

# Reference

[ViT](https://github.com/jeonsworld/ViT-pytorch)
[TransUNet](https://github.com/Beckschen/TransUNet)
[SHOT](https://github.com/tim-learn/SHOT)

# Contributers