# Exercise 7: Representation learning, Explainable AI and Knowledge Extraction

## Overview
This two-part exercise will first guide you through the fundamentals and evaluation of representation learning, and then shift focus to decoding how pre-trained models make decisions using Explainable AI (XAI).

The aim of part A is to explore unsupervised representation learning by building and training a Variational Autoencoder (VAE). We will be working with the classic MNIST dataset. Unlike supervised learning, our model won't be given any true labels to start with; instead, it must learn to compress the images into a compact "latent space" and reconstruct them, discovering the dataset's structure on its own.

In this exercise, we will construct a VAE from scratch and see how the balance between reconstruction loss and regularization shapes smooth, continuous representations. We will then evaluate this latent space using a simple logistic regression classifier to quantify how cleanly the model separates digit classes—without ever being given their labels! Finally, we will explore the VAE's generative power by interpolating between clusters to create completely new images, concluding with UMAP to visualize the high-dimensional manifolds our model has successfully untangled.

The goal of part B is to learn how to probe what a pre-trained classifier has learned about the data it was trained on. We will be working with a simple example which is a fun derivation on the MNIST dataset that you will have seen in previous exercises in this course. 
Unlike regular MNIST, our dataset is classified not by number, but by color! The question is... which colors fall within which class?

![CMNIST](assets/cmnist.png)

In this exercise, we will return to conventional, gradient-based attribution methods to see what they can tell us about what the classifier knows. 
We will see that, even for such a simple problem, there is some information that these methods do not give us. 

We will then train a generative adversarial network, or GAN, to try to create counterfactual images. 
These images are modifications of the originals, which are able to fool the classifier into thinking they come from a different class!. 
We will evaluate this GAN using our classifier; Is it really able to change an image's class in a meaningful way? 

Finally, we will combine the two methods — attribution and counterfactual — to get a full explanation of what exactly it is that the classifier is doing. We will likely learn whether it can teach us anything, and whether we should trust it!

## Setup

Before anything else, in [the super-repository](https://github.com/ai-mbl/AI-MBL-2025) called `AI-MBL-2025`:
```
git pull
git submodule update --init 07_xai
```

Then, if you have any other exercises still running, please save your progress and shut down those kernels.
This is a GPU-hungry exercise so you're going to need all the GPU memory you can get.

Next, run the setup script. It might take a few minutes.
```
cd 07_xai
bash setup.sh
```
This will:
- Create a `conda` environment for this exercise
- Download the data and train the classifier we're learning about
Feel free to have a look at the `setup.sh` script to see the details.


Next, open the exercise notebook!

### Acknowledgments
This notebook was written by Diane Adjavon, Maria Theiss and Anna Foix-Romero with input from
Alex Hillsley, Ed Hirata, Larissa Heinrich, Morgan Schwartz, Ben Salmon, Albert Dominguez, Talley Lambert and Eva de la Serna.
Part B was inspired by a previous version written by Jan Funke and modified by Tri Nguyen, using code from Nils Eckstein.
Part A has been inspired by multiple discussions between Virginie Uhlhmann, Alex Krull, Martin Weigert, Albert Dominguez, Ed Hirata and Anna Foix-Romero.