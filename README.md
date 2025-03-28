## <div align="center">Environment Setup</div>
```
conda env create -f environment.yml
conda activate MVA
```
## <div align="center">Directory upscaling with SwinIR</div>
```
python swinir.py --task real_sr --model_path swin_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq <path/to/image/dir> --scale 4 --tile 400
```
Results stored in `swin_results/`
## <div align="center">Inference a single image with SAHI</div>
```
python sahi_detect.py --image_path pub_test --action all

--path (required)	Path to the input image (for single inference) or dataset folder (for batch processing, e.g., pub_test).
--action (required)	Specifies the type of inference. Defaults to "all". Options: "all" (batch inference pub_test), "single" (process one image).
--workers_per_gpu (optional)	Number of workers per GPU for batch processing. Defaults to 4.
```
Results stored in `runs/predict`
## <div align="center">SAHI Sample Results</div>

![Sample 00007](sample_image/00007.jpg)
![Sample 00007](sample_image/00007.png)

![Sample 00039](sample_image/00039.jpg)
![Sample 00039](sample_image/00039.png)

![Sample 00182](sample_image/00182.jpg)
![Sample 00182](sample_image/00182.png)

---
### <div align="center">Citation</div>
```bibtex
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}
```
```bibtex
@software{obss2021sahi,
  author       = {Akyon, Fatih Cagatay and Cengiz, Cemil and Altinuc, Sinan Onur and Cavusoglu, Devrim and Sahin, Kadir and Eryuksel, Ogulcan},
  title        = {{SAHI: A lightweight vision library for performing large scale object detection and instance segmentation}},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5718950},
  url          = {https://doi.org/10.5281/zenodo.5718950}
}
```