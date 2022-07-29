# Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models

This is the code repository of the following [paper](https://....pdf) to train and perform inference with patch-based diffusion models for image restoration under adverse weather conditions.

"Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
arXiv preprint, 2022.

Currently the repository is being prepared. Code and model checkpoints are going to be included soon.

In the meantime, check out below for some visualizations of our patch-based diffusive image restoration approach.

## Image Desnowing, Deraining \& Dehazing, Raindrop Removal

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Condition</td>
    <td align="center"><b>Restoration Process</td>
    <td align="center"><b>Output</td>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769278-2ab420b3-6e81-4e9d-9d41-3c1bbbae6d7e.png" alt="snow21"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769272-bb3f8e25-f304-4dc0-922e-71c326b0b01e.gif" alt="snow22"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769282-242711d5-e809-45c3-ab89-3e8fabbe1e97.png" alt="snow23"></td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769267-24c7541f-670a-484c-8e44-12c5e95f1e58.png" alt="snow11"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769262-7a9a8236-f12a-4d68-83e0-068b1ebaf1f7.gif" alt="snow12"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769271-08328a77-5452-4bfe-93fd-eccfcc3099c5.png" alt="snow13"></td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181770508-490da62c-2f73-4d4f-9a97-45c8f5f5ff66.png" alt="rh11"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181771830-fcfa649c-1935-4ef3-a990-1a641266caab.gif" alt="rh12"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181770509-24266aa7-e177-455a-bbce-6d43e71acb77.png" alt="rh13"></td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769984-0072cb4e-c5fc-472a-8c57-58eace811521.png" alt="rd11"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181771825-2d23b266-af62-42a7-9649-586b13570c4f.gif" alt="rd12"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769987-54b1ba62-e023-4a97-9d9d-32a644037109.png" alt="rd13"></td>
  </tr>
</table>

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
