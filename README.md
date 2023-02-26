# Learning to Generalize towards Unseen Domains via a Content-Aware Style Invariant Framework for Disease Detection from Chest X-rays [[ArXiv Paper]()]
By Mohammad Zunaed, Md. Aynal Haque, Taufiq Hasan

![](images/proposed_framework.png)

In this work, we present a novel domain-agnostic multi-label chest X-ray thoracic disease detection framework that can perform well on unseen test datasets. This repository contains the [PyTorch](http://pytorch.org/) implementation of our proposed method.

## Prerequisites
- pytorch (1.11.0+)
- albumentations (1.1.0+)
- python (3.9.12+)

The code is developed in the following hardware and OS setting:
- Ubuntu 18.04.3 LTS OS
- GeForce RTX 2080 Ti GPU (4x)
- Intel Core-i9 7920X @ 2.90 GHz CPU

## Prepare Data
- Download the full-size [Standford CheXpert](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and [BRAX](https://physionet.org/content/brax/1.1.0/) datasets.
- Change the dataset paths of the downloaded datasets inside the following files: `/datasets/process_brax.py`, `/datasets/process_chexpert.py`, and `/datasets/process_mimic.py`. Then, run:
```
sh prepare_data.sh
```
- Download the datasets from [here](https://www.kaggle.com/datasets/mohammadzunaed/lung-segmentation-datasets) and place them under `/lung_segmentation_network/datasets/`  for training the lung segmentation network.
```
├── lung_segmentation_network/
    ├── datasets/     
        ├── jsrt/
        ├── jsrt_gt/
        ├── ranzcr_clip/
        ├── ranzcr_clip_gt/			
...
```
- Modify the paths inside `/lung_segmentation_network/train_and_generate_masks.sh` and run the following command to train the lung segmentation network and generate the lung masks for all three CXR datasets.
```
sh train_and_generate_masks.sh
```
- Download the mini-ImageNet dataset from [here](https://www.kaggle.com/datasets/mohammadzunaed/mini-imagenet) and place them under `/datasets/`.
```
├── datasets/   
    ├── mini_imagenet_train/
    ├── brax/
    ├── chexpert/
    ├── mimic/			
...
```
- Run the following command to generate the stratified cross-validation splits.
```
python prepare_cv_splits.py
```

## Training and Testing
Please update the paths and configs in the `cls_configs.py` and `cls_main.py` and run the following command to train the specific models.
```
python cls_main.py
```

For example, to run multi-source domain-generalization training:
```
python cls_main.py --gpu_ids='0,1,2,3' --n_workers=12 --run_configs_list 'baseline_md_chexpert_mimic' 'baseline_md_chexpert_mimic_mask_crop' 'baseline_md_chexpert_mimic_mask_crop_il_srm' 'proposed_md_DA_chexpert_mimic'
```

To evaluate the model on the unseen domain test dataset:
```
python cls_test_unseen.py --run_config='proposed_md_DA_chexpert_mimic' --test_dict_path='./datasets/split_and_test_dicts/brax_test_dict.npy' --gpu_ids='0,1,2,3' --batch_size=160 --image_resize_dim=224 --n_workers=12
```

To evaluate the model on the seen domain test dataset:
```
python cls_test_seen.py --run_config='proposed_md_DA_chexpert_mimic' --test_dict_path='./datasets/split_and_test_dicts/chexpert_mimic_split_info_dict.npy' --gpu_ids='0,1,2,3' --batch_size=160 --image_resize_dim=224 --n_workers=12
```

## Citation
If you use this code in your research please consider citing
```
@article{
}
```
