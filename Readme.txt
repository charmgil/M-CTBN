Introduction:

This package provides a Matlab implementation of Mixtures of Conditional Tree-structured Bayesian Networks (MC) [Hong, Batal, Hauskrecht 2014] that builds mixtures of structured prediction models for multi-label classification.

To train a model, use CTBN/MC/train_MC.m. To use a trained model for prediction, use CTBN/MC/predict_MC.m.

demo.m contains a demonstration script that learns and uses the MC models on the flags dataset [Eduardo, Plastino, Freitas 2013] using 10-fold cross validation.

The package has been written and tested on Matlab R2013a-R2014b.


----

Disclaimer:

This code package can be used for academic purposes only. We do not guarantee that the code is correct, current or complete, and do not provide any technical support. Accordingly, the users are advised to confirm the correctness of the package before making any decisions with it.


----

Reference:

[Hong, Batal, Hauskrecht 2014] C. Hong, I. Batal, and M. Hauskrecht. A mixtures-of-trees framework for multi-label classification. ACM International Conference on Information and Knowledge Management (CIKM 2014), Shanghai, China. November 2014.

[Fan et al. 2008] R. Fan, K. Chang, C. Hsieh, X. Wang, and C. Lin. LIBLINEAR: A Library for Large Linear Classification, Journal of Machine Learning Research 9(2008), 1871-1874. Software available at http://www.csie.ntu.edu.tw/~cjlin/liblinear

[Eduardo, Plastino, Freitas 2013] G., Eduardo, A. Plastino, and A. Freitas. A Genetic Algorithm for Optimizing the Label Ordering in Multi-Label Classifier Chains. IEEE 25th International Conference on Tools with Artificial Intelligence (ICTAI), pp.469-476, 2013.
