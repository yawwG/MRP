# An explainable longitudinal multi-modal fusion model for predicting neoadjuvant therapy response in breast cancer treatment

We developed a Multi-modal Response Prediction (MRP) system for instant prediction in Neoadjuvant Therapy (NAT) in breast cancer treatment. Using diverse clinical data, MRP mimics physician assessments, aided by a multi-task learning module for practicality in multi-center hospitals. Validated across centers and reader studies, MRP matches radiologists' performance and outperforms in predicting pCR in the Pre-NAT phase. Clinical utility assessment shows MRP's potential: reducing treatment toxicity by 35.8% in Pre-NAT and preventing surgeries by 16.7% in Post-NAT, without mispredictions. Our work enhances AI applications in personalized treatment decisions.

## Neoadjuvant Therapy Pathway
(https://github.com/yawwG/MRP/blob/main/figures/clinical_pathway.png)

## Model system
(https://github.com/yawwG/MRP/blob/main/figures/structure.png)

## Environment Setup
Start by [installing PyTorch 1.8.1](https://pytorch.org/get-started/locally/) with the right CUDA version, then clone this repository and install the dependencies.  

```bash
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch
$ pip install git@github.com:yawwG/MRP.git
$ conda env create -f environment.yml
```

## Code Description
This codebase has been developed with python version 3.7, PyTorch version 1.8.1, CUDA 11.1 and pytorch-lightning 1.5.9. 
Example configurations for MG-based and MRI-based classification can be found in the `./configs`. 
All training and testing are done using the `run.py` script. For more documentation, please run: 

```bash 
python run.py --help
```

The preprocessing steps for each dataset (MRI and Mammogram) can be found in `datasets/image_dataset.py`
The dataset using is specified in config.yaml by key("dataset").

### Train and Test
Training the MRI and Mammogram based model for pCR prediction with the following command: 
```bash 
python run.py -c imrrhpc.yaml --train --test
python run.py -c imgrhpc.yaml --train --test
```

### Multimodal contribution analysis
To ensure that the model's interpretability and predictable performance, we explicitly demonstrate the contribution of multi-modalities in the model's training. 
```bash 
python ./utils/contribution.py
```

### Decision curve analysis analysis
We explored in two specific clinical scenarios: personalizing Pre-/Mid-NAT management of non-pCR patients to avoid toxic therapy and optimizing Post-NAT management of pCR patients to reduce unnecessary surgeries. 
```bash 
python ./utils/dca_analysis.py
```


## Contact details
If you have any questions please contact us. 

Email: ritse.mann@radboudumc.nl (Ritse Mann); taotanjs@gmail.com (Tao Tan); y.gao@nki.nl (Yuan Gao)

Links: [Netherlands Cancer Institute](https://www.nki.nl/), [Radboud University Medical Center](https://www.radboudumc.nl/en/patient-care), [Maastricht University](https://www.maastrichtuniversity.nl/nl), [St Joseph’s Healthcare Hamilton](https://www.stjoes.ca/) and [The University of Hong Kong](https://www.hku.hk/) 

<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/NKI.png" width="166.98" height="87.12"/>
<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/RadboudUMC.png" width="231" height="87.12"/>
<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/Maastricht.png" width="237.6" height="87.12"/>  

<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/hku.png" width="104" height="87.12"/>
<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/st joseph's.png" width="104" height="87.12"/>
