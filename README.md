# Mitigating Carbon Footprint for Knowledge Distillation Based Deep Learning Model Compression

* The code is the official implementation of the work "[Mitigating Carbon Footprint for Knowledge Distillation Based Deep Learning Model Compression](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0285668)" in PLOS One. Published: May 15, 2023, DOI: https://doi.org/10.1371/journal.pone.0285668
* Congratulations and great work to Kazi Rafat, Sadia Islam, Abdullah Al Mahfug and Ismail Hossain.
* Special Thanks to the great supervisors Dr. Shafin Rahman and Dr. Nabeel Mohammad.

### Idea

#### KD generates lighter models and typically performs with slightly less accuracy than the heavier teacher model (model accuracy by the teacher model. Although the distillation process makes models deployable on low-resource devices, they were found to consume an exorbitant amount of energy and have a substantial carbon footprint (15 times more carbon compared to the corresponding teacher model). The enormous environmental cost is primarily attributable to the tuning of the hyperparameter, Temperature (τ). In this article, we propose measuring the environmental costs of deep learning work (in terms of GFLOPS in millions, energy consumption in kWh, and CO2 equivalent in grams). In order to create lightweight models with low environmental costs, we propose a straightforward yet effective method for selecting a hyperparameter (τ) using a stochastic approach for each training batch fed into the models. 

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/King-Rafat/STKD_CFMitigation/blob/main/Figures/journal.pone.0285668.g001.PNG)
Figure 1: Illustration of carbon footprints used by different deep models while (a) training on CIFAR 100 (in log scale) and (b-c) inferring on evaluation set. ResNet18 is a deeper model with 11.2M parameters, resulting in higher inference time (4.7 sec.) and CO2 emission (0.087 g). To minimize this, using ResNet18 as a teacher, we train two student models, MobileNetV2 (student 1) and ShuffleNetV2 (student 2), following the traditional KD process. This training costs significant carbon footprints (red and green dashed curves in (a)) with an accuracy increment from learning the teacher model (black dotted curve in (a)). However, as expected, both students consume less time and CO2 during inference (red and green shaded bars in (b) and (c)). We aim to reduce the training cost and CO2 production of the KD process while using the same students (red and green solid curves in (a)) and maintain similar accuracy and inference costs (solid red and green bars in (b) and (c)) in comparison with the costly KD training.

## Results
### Image Classification and Object Detection

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/King-Rafat/STKD_CFMitigation/blob/main/Figures/journal.pone.0285668.t004.PNG)


### Datasets

* CIFAR 10: 

Paper link: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Website: https://www.cs.toronto.edu/~kriz/cifar.html

File: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

* CIFAR 100: 
Paper link: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Website: https://www.cs.toronto.edu/~kriz/cifar.html

File: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

* Tiny ImageNet: 

Paper link: http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf

File: http://cs231n.stanford.edu/tiny-imagenet-200.zip

* Pascal voc 2012:  

Paper link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf

Website: http://host.robots.ox.ac.uk/pascal/VOC/

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

* Pascal voc 2007:  

Paper link: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf

Website: http://host.robots.ox.ac.uk/pascal/VOC/

File: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar, and http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

### Some Pretrained Models:
https://figshare.com/articles/online_resource/Pretrained_Models_for_the_paper_Mitigating_Carbon_Footprint_for_Knowledge_Distillation-Based_Deep_Learning_Model_Compression_accepted_into_PLOS_one_/22761962

### Files

* `STKD.py` : perform KD training, base training, regularization training, stochastic training (Tiny ImageNet custom dataset added). 
* `Experiments` : consists of JSON files for carrying out experiments.
* `model` : Consists of model architectures that can be used.

### How to run
- 

### Used Repositories

* Github Repository for DATA-FREE KNOWLEDGE DISTILLATION: https://github.com/zju-vipa/CMI

* Github Repository for OBJECT DETECTION KNOWLEDGE DISTILLATION: https://github.com/SsisyphusTao/Object-Detection-Knowledge-Distillation/tree/mbv2-lite

### Dependencies
- python>=3.7
- pytorch==1.12.1, torchvision==0.13.1, torchaudio==0.12.1, cudatoolkit=11.3
- carbontracker

<!-- ## Contact and Checkout the Authors:
Sadia Islam: sadia.islam5@northsouth.edu

Abdullah Al Mahfug: abdullah.mahfug@northsouth.edu

Md. Ismail Hossain: ismail.hossain2018@northsouth.edu

Dr. Nabeel Mohammad: nabeel.mohammed@northsouth.edu -->

## Cite This Paper
If you use this code and model and dataset splits for your research, please consider citing:

```
@article{rafat2023mitigating,
  title={Mitigating carbon footprint for knowledge distillation based deep learning model compression},
  author={Rafat, Kazi and Islam, Sadia and Mahfug, Abdullah Al and Hossain, Md Ismail and Rahman, Fuad and Momen, Sifat and Rahman, Shafin and Mohammed, Nabeel},
  journal={Plos one},
  volume={18},
  number={5},
  pages={e0285668},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
