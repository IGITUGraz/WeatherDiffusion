# Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models

This is the code repository of the following [paper](https://....pdf) to train and perform inference with patch-based diffusion models for image restoration under adverse weather conditions.

"Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
arXiv preprint, 2022.

Currently the repository is being prepared. Code and model checkpoints are going to be included soon.

In the meantime, check out below for some visualizations of our patch-based diffusive image restoration approach!

## Image Desnowing


## Image Deraining \& Dehazing


## Removing Raindrops



## Setup

You will need [PyTorch](https://pytorch.org/get-started/) to run this code. You can simply start by executing:
```bash
pip install -r requirements.txt
```
to install all dependencies and use the repository.


## Datasets

We perform experiments for image desnowing on [Snow100K](https://sites.google.com/view/yunfuliu/desnownet), combined image deraining and dehazing on [Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), and raindrop removal on
the [RainDrop](https://github.com/rui1996/DeRaindrop) datasets. To train multi-weather restoration, we used the AllWeather training set from [TransWeather](https://github.com/jeya-maria-jose/TransWeather), which is composed of subsets of training images from these three benchmarks.


## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@article{ozdenizci2022,
  title={Restoring vision in adverse weather conditions with patch-based denoising diffusion models},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  journal={arxiv preprint},
  year={2022}
}
```

## Acknowledgments

Authors of this work are affiliated with Graz University of Technology, Institute of Theoretical Computer Science, and Silicon Austria Labs, TU Graz - SAL Dependable Embedded Systems Lab, Graz, Austria. This work has been supported by the "University SAL Labs" initiative of Silicon Austria Labs (SAL) and its Austrian partner universities for applied fundamental research for electronic based systems.

Parts of this code repository is based on the following works:

* https://github.com/ermongroup/ddim
* https://github.com/bahjat-kawar/ddrm
* https://github.com/JingyunLiang/SwinIR
