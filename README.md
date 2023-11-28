# An explainable longitudinal multi-modal fusion model for predicting neoadjuvant therapy response in breast cancer treatment

We developed a Multi-modal Response Prediction (MRP) system for instant prediction in Neoadjuvant Therapy (NAT) in breast cancer treatment. Using diverse clinical data, MRP mimics physician assessments, aided by a multi-task learning module for practicality in multi-center hospitals. Validated across centers and reader studies, MRP matches radiologists' performance and outperforms in predicting pCR in the Pre-NAT phase. Clinical utility assessment shows MRP's potential: reducing treatment toxicity by 35.8% in Pre-NAT and preventing surgeries by 16.7% in Post-NAT, without mispredictions. Our work enhances AI applications in personalized treatment decisions.

## Neoadjuvant Therapy Pathway
<img src="https://github.com/yawwG/MRP/blob/main/figures/clinical_pathway.png"/>
Fig 1. **a. Pre-NAT phase.** BC is diagnosed following a tumor screening/diagnosis (mammography and/or ultrasound) and biopsy, subsequent histopathology analysis, and a staging breast MRI. These measures help derive demographic, radiological, clinical and histopathological variables describing the patient's state at diagnosis. **b. Mid-NAT phase.** The mid-NAT MRI is performed to assess the response and see if therapy adjustments for unresponsive patients. **c. Post-NAT phase.** Breast MRI is used to assess if patients have achieved pCR. Subsequently, patients undergo surgery, and a histological examination is performed, assessing the Post-NAT TN, which is the gold standard.  

## Model system
<img src="https://github.com/yawwG/MRP/blob/main/figures/structure.png"/>
Fig 2. **a. Model development.** In this work, we developed and evaluated a deep learning system that utilizes multiple modalities to predict the response of breast cancer patients across neoadjuvant therapy (NAT) care. The system incorporates deep neural networks trained on Pre-NAT mammogram images and longitudinal MRI scans through NAT, which were retrospectively gathered. In addition to the imaging data, we collected auxiliary information referred to as rhpc which includes radiological assessments (r), histopathological assessments (h), personal patient records (p), and clinical data (c).
After data retrieval, iMGrhpc and iMRrhpc were modeled independently, where iMGrhpc is based on Pre-NAT mammogram and rhpc data, while iMRrhpc is based on longitudinal MRIs embedding temporal information and rhpc data. These models were further utilized to create an aggregated model called MRP, which aggregates and optimizes the outputs of iMGrhpc and iMRrhpc.
**b. NAT response assessment of AI model and reader study.** To assess the performance of MRP in predicting the pathological response (pCR vs. non-pCR) at different stages, including Pre-NAT (before administration of NAT), Mid-NAT (during therapy), and Post-NAT (prior to surgery), we validated the model using standard metrics: AUROC (Area Under Receiver Operating Characteristic Curve) and AUPRC (Area Under Precision-Recall Curve). In order to compare the performance of MRP with human experts, we conducted a reader study involving six international breast radiologists. The readers interpreted a randomly selected subset of MRI examinations and provided their predictions of the probability of patient response to therapy. 
**c. External validation.** Additionally, we evaluated our system on an external dataset obtained from Duke University (288 patients) to assess its generalizability. 
**d. Personalizing management in clinical practice.** Furthermore, we simulated two scenarios to assess the system's ability to identify non-pCR patients before neoadjuvant therapy in whom toxic treatments may be timely adapted, and to identify pCR (ypT0) patients before surgery who might not need surgical procedures. This simulation utilized a decision curve analysis (DCA) methodology. Circled C stands for current clinical practice; Circled AI stands for our MRP system suggested strategy.

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
<div align="center">
<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/NKI.png" width="166.98" height="87.12"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/RadboudUMC.png" width="231" height="87.12"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/Maastricht.png" width="237.6" height="87.12"/>  
 </div>
 <div align="center">
<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/hku.png" width="104" height="87.12"/> <img src="https://github.com/yawwG/MRP/blob/main/figures/st_joseph's.png" width="100" height="87.12"/>
 </div>
