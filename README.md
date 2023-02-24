### Learning to Generalize towards Unseen Domains via a Content-Aware Style Invariant Framework for Disease Detection from Chest X-rays [[ArXiv Paper]()]
By Mohammad Zunaed, Md. Aynal Haque, Taufiq Hasan

![](images/proposed_framework.png)

# Prepare Data
- Download the full-size [Standford CheXpert](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and [BRAX](https://physionet.org/content/brax/1.1.0/) datasets.
- Change the dataset paths of the downloaded datasets inside the following files: /datasets/process_brax.py, /datasets/process_chexpert.py, and /datasets/process_mimic.py. Then, run:
```
sh prepare_data.sh
```
- Download the datasets from [here]() and place them under /lung_segmentation_network/datasets/  for training the lung segmentation network.
```
├── lung_segmentation_network/
    ├── datasets/     
        ├── jsrt/
        ├── jsrt_gt/
        ├── ranzcr_clip/
        ├── ranzcr_clip_gt/			
...
```
- Modify the paths inside /lung_segmentation_network/train_and_generate_masks.sh and run the following command to train the lung segmentation network and generate the lung masks for all three CXR datasets.
```
sh train_and_generate_masks.sh
```
- Download the mini-ImageNet dataset from [here]() and place them under /datasets/.
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

### Citation
If you use this code in your research please consider citing
```
@article{luo2021category,
  title={Category-Level Adversarial Adaptation for Semantic Segmentation using Purified Features},
  author={Luo, Yawei and Liu, Ping and Zheng, Liang and Guan, Tao and Yu, Junqing and Yang, Yi},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence (TPAMI)},
  year={2021},
}

@inproceedings{luo2019Taking,
title={Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation},
author={Luo, Yawei and Zheng, Liang and Guan, Tao and Yu, Junqing and Yang, Yi},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2019}
}
```
