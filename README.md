# SceneClassifier

Written Winter 2016

This was submitted as part of the Computer Vision Coursework, in which we were tasked to develop a scene classifier using
the OpenIMAJ Library. The Summary of the Runs are:

Run #1: We developed a simple k-nearest-neighbour classifier using the “tiny image” feature by cropping each image to a square about the centre, and then resizes it to a small, fixed resolution (we recommend 16x16). The pixel values can be packed into a vector by concatenating each image row.

Run #2: A set of linear classifiers was developed (use the LiblinearAnnotator class to automatically create 15 one-vs-all classifiers) using a bag-of-visual-words feature based on fixed size densely-sampled pixel patches.We used 8x8 patches, sampled every 4 pixels in the x and y directions. A sample of these was then clustered using K-Means to learn a vocabulary.

Run #3: You should try to develop the best classifier you can! You can choose whatever feature, encoding and classifier you like. Potential features: the GIST feature; Dense SIFT; Dense SIFT in a Gaussian Pyramid; Dense SIFT with spatial pooling (i.e. PHOW as in the OpenIMAJ tutorial), etc. Potential classifiers: Naive bayes; non-linear SVM (perhaps using a linear classifier with a Homogeneous Kernel Map)
