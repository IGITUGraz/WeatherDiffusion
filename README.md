# Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models

This is the code repository of the following [paper](https://arxiv.org/pdf/2207.14626.pdf) to train and perform inference with patch-based diffusion models for image restoration under adverse weather conditions.

"Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023.\
https://doi.org/10.1109/TPAMI.2023.3238179

## Datasets

We perform experiments for image desnowing on [Snow100K](https://sites.google.com/view/yunfuliu/desnownet), combined image deraining and dehazing on [Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), and raindrop removal on
the [RainDrop](https://github.com/rui1996/DeRaindrop) datasets. To train multi-weather restoration, we used the AllWeather training set from [TransWeather](https://github.com/jeya-maria-jose/TransWeather), which is composed of subsets of training images from these three benchmarks.


## Saved Model Weights

We share a pre-trained diffusive **multi-weather** restoration model [WeatherDiff<sub>64</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar) with the network configuration in `configs/allweather.yml`.
To evaluate WeatherDiff<sub>64</sub> using the pre-trained model checkpoint with the current version of the repository:
```bash
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'raindrop' --sampling_timesteps 25 --grid_r 16
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'rainfog' --sampling_timesteps 25 --grid_r 16
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'snow' --sampling_timesteps 25 --grid_r 16
```

A smaller value for `grid_r` will yield slightly better results and higher image quality:
```bash
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'raindrop' --sampling_timesteps 25 --grid_r 4
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'rainfog' --sampling_timesteps 25 --grid_r 4
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'snow' --sampling_timesteps 25 --grid_r 4
```

We also share our pre-trained diffusive multi-weather restoration model [WeatherDiff<sub>128</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar) with the network configuration in `configs/allweather128.yml`.

Check out below for some visualizations of our patch-based diffusive image restoration approach.

## Image Desnowing

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Condition</td>
    <td align="center"><b>Restoration Process</td>
    <td align="center"><b>Output</td>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769278-2ab420b3-6e81-4e9d-9d41-3c1bbbae6d7e.png" alt="snow11"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351181-9528c4cb-218d-4b06-8c4c-210219ace8bc.gif" alt="snow12"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769282-242711d5-e809-45c3-ab89-3e8fabbe1e97.png" alt="snow13"></td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769267-24c7541f-670a-484c-8e44-12c5e95f1e58.png" alt="snow21"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351179-b0183145-ce70-4ded-87eb-077a22c9112a.gif" alt="snow22"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769271-08328a77-5452-4bfe-93fd-eccfcc3099c5.png" alt="snow23"></td>
  </tr>
</table>
  
## Image Deraining \& Dehazing

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Condition</td>
    <td align="center"><b>Restoration Process</td>
    <td align="center"><b>Output</td>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181770508-490da62c-2f73-4d4f-9a97-45c8f5f5ff66.png" alt="rh11"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351163-7913703b-977f-4117-95ce-2e88397be6be.gif" alt="rh12"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181770509-24266aa7-e177-455a-bbce-6d43e71acb77.png" alt="rh13"></td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351171-fd874818-d797-409a-9988-28824091417f.png" alt="rh21"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351167-94807242-a5ba-473e-8503-11f9c294b9bf.gif" alt="rh22"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351176-a9f49787-e7ed-45bc-b9ac-d6585a81bd09.png" alt="rh23"></td>
  </tr>
</table>

## Raindrop Removal

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Condition</td>
    <td align="center"><b>Restoration Process</td>
    <td align="center"><b>Output</td>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769984-0072cb4e-c5fc-472a-8c57-58eace811521.png" alt="rd11"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351153-785519aa-3df2-4141-89f3-c8837345eeb3.gif" alt="rd12"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/181769987-54b1ba62-e023-4a97-9d9d-32a644037109.png" alt="rd13"></td>
  </tr>
  <tr>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351159-e9953ae1-652a-4bdd-a254-6ba823e5444d.png" alt="rd21"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351158-5ccb3215-5d52-4cda-8dcf-825629fb9f1c.gif" alt="rd22"></td>
    <td> <img src="https://user-images.githubusercontent.com/30931390/182351162-1251b9bb-da71-4d1a-9a11-a4ee5729e1b2.png" alt="rd23"></td>
  </tr>
</table>


## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@article{ozdenizci2023,
  title={Restoring vision in adverse weather conditions with patch-based denoising diffusion models},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  pages={1-12},
  year={2023},
  doi={10.1109/TPAMI.2023.3238179}
}
```

## Acknowledgments

Authors of this work are affiliated with Graz University of Technology, Institute of Theoretical Computer Science, and Silicon Austria Labs, TU Graz - SAL Dependable Embedded Systems Lab, Graz, Austria. This work has been supported by the "University SAL Labs" initiative of Silicon Austria Labs (SAL) and its Austrian partner universities for applied fundamental research for electronic based systems.

Parts of this code repository is based on the following works:

* https://github.com/ermongroup/ddim
* https://github.com/bahjat-kawar/ddrm
* https://github.com/JingyunLiang/SwinIR
