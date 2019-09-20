# Emotion Recognition Papers

list of Emotion Recognition Papers

## Facial expression

| Paper                                                        | metric                                          | citation | code link                                                    | note                                       |
| ------------------------------------------------------------ | ----------------------------------------------- | -------- | ------------------------------------------------------------ | ------------------------------------------ |
| Real-time Convolutional Neural Networks for Emotion and Gender Classification(2017) | 66% in FER-2013                                 | 43       | [oarriaga/face_classification](https://github.com/oarriaga/face_classification) |                                            |
| A Micro-GA Embedded PSO Feature Selection Approach to Intelligent Facial Emotion Recognition(2016) | 100% within CK+ and 94.66% on MMI(cross-domain) | 89       |                                                              | hand craft feature                         |
| Intelligent facial emotion recognition based on stationary wavelet entropy and Jaya algorithm(2018) | 96.80 ± 0.14% on self-made dataset(7 classes)   | 56       |                                                              |                                            |
| Multi-Objective Based Spatio-Temporal Feature Representation Learning Robust to Expression Intensity Variations for Facial Expression Recognition(2017) | 78%，60.98% on MMI，CASME II                    | 32       |                                                              | combined  temporal and spatial information |
| Softmax regression based deep sparse autoencoder network for facial emotion recognition in human-robot interaction(2018) | 89% on both JAFFE and CK+                       | 24       |                                                              | adopted on a robot                         |
| A Study on Emotion Recognition Based on Hierarchical Adaboost Multi-class Algorithm(2018) | 93% on CFAPS                                    | 1        |                                                              | Adaboost                                   |
| Multi-Objective Differential Evolution for feature selection in Facial Expression Recognition systems(2017) | 98.37% on CK+, 92.75% on JAFFE, 84.07% on MMI   | 22       |                                                              | SVM                                        |



### code list

| code                                                         | star | dataset                                                      |                                                              | paper                                                        | performance |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------- |
| [d-acharya/CovPoolFER](https://github.com/d-acharya/CovPoolFER) | 50   | Real-World Affective Faces([RAF-DB](http://www.whdeng.cn/RAF/model1.html)) | 29672 images6 basic expressions plus neutral and 12 compound expressions | [Covariance Pooling For Facial Expression Recognition](https://arxiv.org/abs/1805.04855)(cited 15)（[Covariance Pooling](https://paperswithcode.com/sota/facial-expression-recognition-on-static)） | 87.0%       |
| [d-acharya/CovPoolFER](https://github.com/d-acharya/CovPoolFER) | 50   | [Static Facial Expressions in the Wild](https://paperswithcode.com/sota/facial-expression-recognition-on-static) |                                                              | [Covariance Pooling For Facial Expression Recognition](https://arxiv.org/abs/1805.04855)(cited 15)（[Covariance Pooling](https://paperswithcode.com/sota/facial-expression-recognition-on-static)） | 58.14%      |
| [ bbenligiray/greedy-face-features](https://github.com/bbenligiray/greedy-face-features) | 8    | Cohn-Kanade                                                  |                                                              | [Greedy Search for Descriptive Spatial Face Features](https://ieeexplore.ieee.org/document/7952406)(cited 3)（Sequential forward selection） | 88.7%       |
| [MaxLikesMath/DeepLearningImp](https://github.com/MaxLikesMath/DeepLearningImplementations) | 5    | [MMI](https://paperswithcode.com/sota/facial-expression-recognition-on-mmi) | 740 images and 2,900 videos6 basic expressions + neutral     | [DeXpression: Deep Convolutional Neural Network for Expression Recognition](https://arxiv.org/abs/1509.05371)(cited 74) （DeXpression） | 98.63%      |
| [Open-Debin/Emotion-FAN](https://github.com/Open-Debin/Emotion-FAN) | 9    | [The Extended Cohn-Kanade Dataset(CK+)](https://paperswithcode.com/sota/facial-expression-recognition-on-the-extended) |                                                              | [Frame attention networks for facial expression recognition in videos](https://arxiv.org/abs/1907.00193) | 99.69%      |
| [Open-Debin/Emotion-FAN](https://github.com/Open-Debin/Emotion-FAN) | 9    | [Acted Facial Expressions In The Wild (AFEW)](https://paperswithcode.com/sota/facial-expression-recognition-on-acted-facial) | 1,809 videos6 basic expressions + neutral                    | [Frame attention networks for facial expression recognition in videos](https://arxiv.org/abs/1907.00193) | 51.181%     |



FER systems can be divided into two main categories according to the feature representations: static image FER and dynamic sequence FER. 

### Static image FER

| Dataset  | Paper                                                        | metric                                              | citation | code                                                         | Arch           | Note      |
| -------- | ------------------------------------------------------------ | --------------------------------------------------- | -------- | ------------------------------------------------------------ | -------------- | --------- |
| CK+      | [Facial expression recognition via learning deep sparse autoencoders](https://www.sciencedirect.com/science/article/pii/S0925231217314649) | 7 classes : 95.79 (93.78)  8 classes: 89.84 (86.82) | 101      |                                                              | DAE (DSAE)     | 2018      |
| CK+      | [From facial expression recognition to interpersonal relation prediction](https://arxiv.org/abs/1609.06426) | 6 classes: 98.9                                     | 24       |                                                              | CNN            | 2018      |
| CK+      | [Facial expression recognition by deexpression residue learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Facial_Expression_Recognition_CVPR_2018_paper.pdf) | 7 classes : 97.30 (96.57)                           | 33       |                                                              | GAN (cGAN)     | CVPR2018  |
|          |                                                              |                                                     |          |                                                              |                |           |
| MMI      | [Facial expression recognition by deexpression residue learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Facial_Expression_Recognition_CVPR_2018_paper.pdf) | 6 classes: 73.23 (72.67)                            | 33       |                                                              | GAN (cGAN)     | 2018      |
| MMI      | [Adaptive deep metric learning for identity-aware facial expression recognition](https://ieeexplore.ieee.org/document/8014813) | 6 classes: 78.53 (73.50)                            | 41       |                                                              | CNN            | 2017      |
| MMI      | [Reliable crowdsourcing and deep localitypreserving learning for expression recognition in the wild](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf) | 6 classes: 78.46                                    | 97       |                                                              | CNN            | 2017      |
|          |                                                              |                                                     |          |                                                              |                |           |
| TFD      | [Facenet2expnet: Regularizing a deep face recognition net for expression recognition](https://arxiv.org/abs/1609.06591) | Test: 88.9 (87.7)                                   | 104      |                                                              | CNN(fine-tune) | 2017      |
|          |                                                              |                                                     |          |                                                              |                |           |
| SFEW 2.0 | [Facenet2expnet: Regularizing a deep face recognition net for expression recognition](https://arxiv.org/abs/1609.06591) | Validation: 55.15 (46.6)                            | 104      |                                                              | CNN(fine-tune) | 2017      |
|          |                                                              |                                                     |          |                                                              |                |           |
| CAER     | [Context-Aware Emotion Recognition Networks](https://arxiv.org/abs/1908.05913) | 73.51                                               | 0        |                                                              |                | ICCV 2019 |
|          |                                                              |                                                     |          |                                                              |                |           |
| 未知     | [Joint Pose and Expression Modeling for Facial Expression Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Joint_Pose_and_CVPR_2018_paper.pdf) |                                                     | 23       | [Facial-expression-recognition](https://github.com/FFZhang1231/Facial-expression-recognition) |                | CVPR 2018 |
|          |                                                              |                                                     |          |                                                              |                |           |



### Dynamic image FER

| Dataset   | Paper                                                        | metric                           | citation | code | Arch              | Note      |
| --------- | ------------------------------------------------------------ | -------------------------------- | -------- | ---- | ----------------- | --------- |
| CK+       | [A compact deep learning model for robust facial expression recognition](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Kuo_A_Compact_Deep_CVPR_2018_paper.pdf) | 7 classes: 98.47                 | 12       |      | IntraFace         | CVPR2018  |
| CK+       | [Facial expression recognition based on deep evolutional spatial-temporal networks](https://ieeexplore.ieee.org/document/7890464) | 7 classes: 98.50 (97.78)         | 92       |      | SDM/ Cascaded CNN | 2017      |
|           |                                                              |                                  |          |      |                   |           |
| MMI       | [Deep spatial-temporal feature fusion for facial expression recognition in static images](https://www.sciencedirect.com/science/article/pii/S0167865517303902) | 6 classes: 91.46                 | 6        |      |                   | 2017      |
|           |                                                              |                                  |          |      |                   |           |
| OuluCASIA | [A compact deep learning model for robust facial expression recognition]() | 6 classes: 91.67                 | 12       |      | IntraFace         | CVPR2018  |
| OuluCASIA | [Mode Variational LSTM Robust to Unseen Modes of Variation: Application to Facial Expression Recognition](https://arxiv.org/pdf/1811.06937.pdf) | 85.18(**AFEW:51.44 MPMI:84.98**) | 0        |      | LSTM              | AAAI2019  |
|           |                                                              |                                  |          |      |                   |           |
| CAER      | [Context-Aware Emotion Recognition Networks](https://arxiv.org/abs/1908.05913) | 77.04                            | 0        |      |                   | ICCV 2019 |



## Speech

| Paper                                                        | metric | citation | code link | note |
| ------------------------------------------------------------ | ------ | -------- | --------- | ---- |
| Adieu features? End-to-end speech emotion recognition using a deep convolutional recurrent network（2016,ICASSP） |        | 315      |           |      |

## Pose

under going...

## Signal

| Paper                                              | metric           | citation | code link | note |
| -------------------------------------------------- | ---------------- | -------- | --------- | ---- |
| Emotion recognition using wireless signals（2016） | 85% in 4 classes | 146      |           |      |

## Multimodal

| Paper                                                        | metric                          | citation | code link | note |
| ------------------------------------------------------------ | ------------------------------- | -------- | --------- | ---- |
| EmoNets: Multimodal deep learning approaches for emotion recognition in video(2014) | 47.67% in 2013 EmotiW challenge | 146      |           |      |
| Fusion of Facial Expressions and EEG for Multimodal Emotion Recognition(2017) | 81.25%                          | 13       |           |      |