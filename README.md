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


### Used Repositories

* Github Repository for DATA-FREE KNOWLEDGE DISTILLATION: https://github.com/zju-vipa/CMI

* Github Repository for OBJECT DETECTION KNOWLEDGE DISTILLATION: https://github.com/SsisyphusTao/Object-Detection-Knowledge-Distillation/tree/mbv2-lite

<!-- ## Contact and Checkout the Authors:
Sadia Islam: sadia.islam5@northsouth.edu

Abdullah Al Mahfug: abdullah.mahfug@northsouth.edu

Md. Ismail Hossain: ismail.hossain2018@northsouth.edu

Dr. Nabeel Mohammad: nabeel.mohammed@northsouth.edu -->

## Cite This Paper
If you use this code and model and dataset splits for your research, please consider citing:

```
@article{10.1371/journal.pone.0285668,
    doi = {10.1371/journal.pone.0285668},
    author = {Rafat, Kazi AND Islam, Sadia AND Mahfug, Abdullah Al AND Hossain, Md. Ismail AND Rahman, Fuad AND Momen, Sifat AND Rahman, Shafin AND Mohammed, Nabeel},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Mitigating carbon footprint for knowledge distillation based deep learning model compression},
    year = {2023},
    month = {05},
    volume = {18},
    url = {https://doi.org/10.1371/journal.pone.0285668},
    pages = {1-22},
    abstract = {Deep learning techniques have recently demonstrated remarkable success in numerous domains. Typically, the success of these deep learning models is measured in terms of performance metrics such as accuracy and mean average precision (mAP). Generally, a model’s high performance is highly valued, but it frequently comes at the expense of substantial energy costs and carbon footprint emissions during the model building step. Massive emission of CO2 has a deleterious impact on life on earth in general and is a serious ethical concern that is largely ignored in deep learning research. In this article, we mainly focus on environmental costs and the means of mitigating carbon footprints in deep learning models, with a particular focus on models created using knowledge distillation (KD). Deep learning models typically contain a large number of parameters, resulting in a ‘heavy’ model. A heavy model scores high on performance metrics but is incompatible with mobile and edge computing devices. Model compression techniques such as knowledge distillation enable the creation of lightweight, deployable models for these low-resource devices. KD generates lighter models and typically performs with slightly less accuracy than the heavier teacher model (model accuracy by the teacher model on CIFAR 10, CIFAR 100, and TinyImageNet is 95.04%, 76.03%, and 63.39%; model accuracy by KD is 91.78%, 69.7%, and 60.49%). Although the distillation process makes models deployable on low-resource devices, they were found to consume an exorbitant amount of energy and have a substantial carbon footprint (15.8, 17.9, and 13.5 times more carbon compared to the corresponding teacher model). The enormous environmental cost is primarily attributable to the tuning of the hyperparameter, Temperature (τ). In this article, we propose measuring the environmental costs of deep learning work (in terms of GFLOPS in millions, energy consumption in kWh, and CO2 equivalent in grams). In order to create lightweight models with low environmental costs, we propose a straightforward yet effective method for selecting a hyperparameter (τ) using a stochastic approach for each training batch fed into the models. We applied knowledge distillation (including its data-free variant) to problems involving image classification and object detection. To evaluate the robustness of our method, we ran experiments on various datasets (CIFAR 10, CIFAR 100, Tiny ImageNet, and PASCAL VOC) and models (ResNet18, MobileNetV2, Wrn-40-2). Our novel approach reduces the environmental costs by a large margin by eliminating the requirement of expensive hyperparameter tuning without sacrificing performance. Empirical results on the CIFAR 10 dataset show that the stochastic technique achieves an accuracy of 91.67%, whereas tuning achieves an accuracy of 91.78%—however, the stochastic approach reduces the energy consumption and CO2 equivalent each by a factor of 19. Similar results have been obtained with CIFAR 100 and TinyImageNet dataset. This pattern is also observed in object detection classification on the PASCAL VOC dataset, where the tuning technique performs similarly to the stochastic technique, with a difference of 0.03% mAP favoring the stochastic technique while reducing the energy consumptions and CO2 emission each by a factor of 18.5.},
    number = {5},

}
