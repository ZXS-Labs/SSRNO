# SSRNO
This repository contains the official implementation for Scalable Super Resolution Neural Operator in ACM MM 2024.
## Installation
1. Install requirements from conda
    ```bash
    conda env create -f requirements.yml
    ```
2. Clone the repo
   ```bash
   git clone https://github.com/ZXS-Labs/SSRNO.git
   ```
3. install the modified TorchIntegral
   ```bash
   pip install ./_TO
   ```
4. [optional] install Inplace_Gelu from [Tempo](https://github.com/UofT-EcoSystem/Tempo.git)
   ```bash
    git clone https://github.com/UofT-EcoSystem/Tempo.git
    cd Tempo
    python setup.py install
   ```
## Quick start
Download the pretrained model from [here](https://drive.google.com/drive/folders/1aL0gS2bMVkcPHh5BpqphkzVogF8wVfGA?usp=drive_link).

Link your data folder(s) to `./data/` and change the `root_path` in yaml.
```bash
#training
python inn_attention_train.py \
    --config ./configs/train_ssrno.yaml

#testing
python inn_attention_test.py \
    --config ./configs/test_ssrno.yaml \
    --model "the model pth" \
    --mcell True
```
## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{
    han2024scalable,
    title={Scalable Super-Resolution Neural Operator},
    author={Lei Han and Xuesong Zhang},
    booktitle={ACM Multimedia 2024},
    year={2024},
    url={https://openreview.net/forum?id=COlygxQAV9}
}
```
## Acknowledgements
This code is built on [TorchIntegral](https://github.com/TheStageAI/TorchIntegral.git)
,[Tempo](https://github.com/UofT-EcoSystem/Tempo.git) and [SRNO](https://github.com/2y7c3/Super-Resolution-Neural-Operator.git).
