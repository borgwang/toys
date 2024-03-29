### mnist-one-shot

Experiment code on training a MNIST classifier with only 10 samples. See my blog post (in Chinese) [here](https://borgwang.github.io/dl/2019/11/29/mnist-one-shot.html).

#### Results

- Baseline

| 模型/准确率 | 随机森林 | GBDT   | SVM        | LogisticRegression | KNN (k=1)  |
| ----------- | -------- | ------ | ---------- | ------------------ | ---------- |
| seed=31     | 0.3978   | 0.2869 | 0.3957     | 0.4145             | 0.3975     |
| seed=317    | 0.4446   | 0.2830 | 0.4406     | 0.4479             | 0.4406     |
| seed=31731  | 0.3894   | 0.2138 | 0.4296     | 0.4227             | 0.4296     |
| Average     | 0.4106   | 0.2612 | 0.4220     | **0.4284**         | 0.4226     |


|            | LeNet with Dropout |
| ---------- | ------------------ |
| seed=31    | 0.4213             |
| seed=317   | 0.4315             |
| seed=31731 | 0.4354             |
| Average    | 0.4294             |

- Data Augmentation (1024)

|            | LeNet with data_augmentation (1024) |
| ---------- | ----------------------------------- |
| seed=31    | 0.5490                              |
| seed=317   | 0.5612                              |
| seed=31731 | 0.5301                              |
| Average    | 0.5467                              |


- Data Augmentation + Samples generated by CGAN

|            | gan_raitio=0.1    | gan_raitio=0.2 | gan_raitio=0.4 | gan_raitio=0.6 | gan_raitio=1.0 |
| ---------- | ----------------- | -------------- | -------------- | -------------- | -------------- |
| seed=31    | 0.5844            | 0.5569         | 0.5755         | 0.5598         | 0.5030         |
| seed=317   | 0.5552            | 0.5901         | 0.5424         | 0.5180         | 0.5573         |
| seed=31731 | 0.5828            | 0.5452         | 0.5612         | 0.5469         | 0.5627         |
| Average    | 0.5741            | 0.5640         | 0.5579         | 0.5415         | 0.5410         |

- Data Augmentation + Samples generated by CGAN + Transfer Learning (pretrained ResNet)

|            | data_augment (1024) + cgan_augment (gan_ratio=0.1) + pretrained ResNet-34 |
| ---------- | ------------------------------------------------------------ |
| seed=31    | 0.6823                                                       |
| seed=317   | 0.7265                                                       |
| seed=31731 | 0.6289                                                       |
| Average    | 0.6792                                                       |

