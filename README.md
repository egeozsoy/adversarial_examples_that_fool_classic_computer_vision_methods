# Bachelorthesis: Adversarial Examples That Fool Classic Computer Vision Methods

#### This repository contains my bachelor thesis I did at TUM. Among other things, it includes:

- Training many different machine learning models with scikit-learn
- Use of opencv for extracting sift features
- Extracting bag of visual words features, HOG features
- Use of fishervector
- Use of keras with scikit-learn

------------------------------------------------------------------------------------------------------------------------------

## Abstract
With the recent advancements in deep learning, image recognition started to play a critical role in many different tasks. Every day, it is used more and more for security-critical applications, such as face authentication and autonomous driving. It is well known that deep learning based approaches can be fooled using adversarial examples, images with small perturbations that are classified falsely. Since their discovery, the research community has been working on ways to generate better such examples, and also better defend against them. One area that was not investigated thoroughly is how robust the image recognition methods that are based on classical computer vision methods are. These are algorithms that use handcrafted feature extractors in conjunction with classical machine learning models. This study compares three commonly used feature extractors, BoVW (Bag of Visual Words), Fisher Vector, and HOG (Histogram of Oriented Gradients); with three classical machine learning classifiers, Support Vector Machine, Logistic Regression, and Random Forest, and concludes that all image recognition methods are susceptible to adversarial attacks, albeit in different degrees. BoVW and Fisher Vector with Support Vector Machine or Logistic Regression, are generally the most robust, while CNN (Convolutional Neural Network) and HOG based approaches are generally the least robust. Nonetheless, deep learning only exaggerated the problem of adversarial examples and did not start it, as adversarial examples are observed for any image recognition method tested in this thesis.


