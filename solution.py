# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,title,-all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: 07_xai
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Exercise 7: Representation learning and XAI
#
# Representation learning has been gaining significant attention over the past years,
# driven by the rise of new architectures and the growing need in biology to extract meaningful structured information
# from increasingly high-dimensional and complex data. Yet as these models become more capable, understanding what they are actually learning becomes
# just as important as their predictive performance, and that is precisely the role of explainable AI.
#
# ### Goal of the exercises
#
# The goal of this exercise is to first build an understanding of what representation learning is,
# what we mean by a "representation" in that context, and what makes a representation good or useful.
# From there, we will explore what architectures can be used to obtain these representations, and how we can evaluate
# the quality of the obtained representations.  The second half of the exercise shifts focus to Explainable AI (XAI),
# where the goal is to learn how to probe what a pre-trained classifier has learned about the data it was trained on.
#
# In part A, we will be building models for representation learning
#
# We will:
# 1. Build and train a variational autoencoder (VAE)
# 2. Visualize and compare the latent space learned under different model parameters
# 3. Evaluate representation quality
# 4. Explore the generative properties of VAEs.
#
# In part B, we will be working with a simple example which is a fun derivation on the MNIST dataset that you will have seen in previous exercises in this course.
# Unlike regular MNIST, our dataset is classified not by number, but by color!
#
# We will:
# 1. Load a pre-trained classifier and try applying conventional attribution methods
# 2. Train a GAN to create counterfactual images - translating images from one class to another
# 3. Evaluate the GAN - see how good it is at fooling the classifier
# 4. Create attributions from the counterfactual, and learn the differences between the classes.
#
# If time permits, we will try to apply this all over again as a bonus exercise to a much more complex and more biologically relevant problem.
#
# ### Acknowledgments
#
# This notebook was written by Diane Adjavon, Maria Theiss and Anna Foix-Romero with input from
# Alex Hillsley, Ed Hirata, Larissa Heinrich, Morgan Schwartz, Anna Foix-Romero, Ben Salmon, Albert Dominguez, Talley Lambert and Eva de la Serna.
# Part B was inspired by a previous version written by Jan Funke and modified by Tri Nguyen, using code from Nils Eckstein.
# Part A has been inspired by multiple discussions between Virginie Uhlhmann, Alex Krull, Martin Weigert,
# Albert Dominguez, Ed Hirata and Anna Foix-Romero.
#
# ### AI Statement
#
# Portions of this notebook were developed with the assistance of **Claude 4.6** (Anthropic),
# accessed via the **Harvard AI Sandbox**. Prompts and responses are not used to train external models and data does not leave the
# university's controlled infrastructure.
#
# Specifically, Claude was used for:
# - **Inspiration and structure** of explanatory markdown text
# - **Debugging and refinement** of plotting and training code
# - **Drafting** of reusable helper functions
#
# All AI-assisted content was reviewed, edited, and validated by the notebook authors.
#
# ***
# %% [markdown]
# <div class="alert alert-danger">
# Set your python kernel to <code>07_xai</code>
# </div>
#
# %% [markdown]
# # PART A: Representation learning
# ## Part A.0: Conceptual introduction
# ### What is a representation and why is that useful?
# A representation is a mapping from raw data to a structured, typically lower-dimensional space, called latent space, that ideally captures meaningful features of the data.
#
# Representations
# - Can compress the data
# - Ideally learn meaningful features and discard noise and redundancy
# - Enable classification, clustering, and data generation
# - Aid interpretability
#
# ### What is a good representation?
# In many cases, good representations are...
#
# **Compact**
# Later, we will see example images of MNIST – 28 x 28 images of hand-written digits on dark background.
# Each image provides 28 x 28 = 784 pixel values. However, most of these values carry irrelevant information.
# For instance, most pixels belong to the dark background that does not contain infromation about the imaged digit.
# By keeping a representation compact (but not too compact!), we force the model to discard irrelevant information.
#
# **Smooth and continuous**
# Small changes in the input should lead to small changes in the representation. Ideally, there should be no "gaps" in the representation (this is the case for probabilistic models like Variational AutoEncoders).
# This property makes it possible to generate new data from the representation, as each location in the latent space contains meaningful information.
#
# **Structured**
# Similar inputs should be mapped to close-by locations in latent space, whereas dissimilar inputs should be distant to each other in latent space.
# This means that clusters present in the input, should also appear in latent space.
#
# **Disentangled**
# Ideally, each dimension of the latent space should capture a different feature of the input-image. For example, one dimension could capture the tilt (left-right) of handwritten digits,
# another the identity of the digits, yet another the boldness of the text.
#
# ### Unsupervised learning
# The Variational Autoencoder (VAE) trained in part A is an example of representation learning and unsupervised learning. We do not provide labels – i.e. the model trains on images of hand-written digits, but does not know the true identity (labels 0 - 9) of each image.
# This means the model is learning structural information that is intrinsic to the data. It does so by fulfilling two training objectives:
# - Reconstruction: Reconstruction-loss incentivises the model to generate a reconstructed version of an input-image from the latent space. This means that the latent space needs to ideally carry information to allow for a close reconstruction.
# - Constraints on the latent space: We can impose structural constraints on the latent space, ensuring previously discussed smoothness and continuity.
#
# Unsupervised learning is valuable in applications where labels are costly to generate or entirely unknown.
#
# ---
# %% [markdown]
#
# ## Part A.1: General set-up
# In this part of the notebook, we will load the same dataset as in the previous exercise.
#
# ### Background - the MNIST dataset
# MNIST is a machine learning benchmark dataset:
# * **70,000** grayscale images of handwritten digits **0 - 9**.
# * Of which are **60,000** training images and **10,000** testing images.
# *  Each image has a resolution of **28x28** pixels.
#
# It is a great dataset to introduce representation learning because it is simple enough to train quickly,
# but still structured enough that we can visually inspect and intuitively evaluate the quality of the learned representations
# and reconstructions.
#
# Documentation for this pytorch dataset is available at https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
#
# Let's get started and load our dataset, transforming the images into torch tensors and normalising them.

# %% [markdown]
# ### Load MNIST

# %%
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_mnist = torchvision.datasets.MNIST("./mnist", train=True, download=True, transform=transform)
test_mnist = torchvision.datasets.MNIST("./mnist", train=False, download=True, transform=transform)

# %% [markdown]
# <div class="alert alert-info">
#     <b>Note:</b> set the <code>download</code> argument of <code>torchvision.datasets.MNIST</code> to <code>True</code> or <code>False</code> as required when re-running the notebook. <br>
#     When <code>./mnist</code> does not yet exist (on first run), make sure the first call to <code>torchvision.datasets.MNIST</code> has <code>download</code> set to <code>True</code>.
# </div>

# %% [markdown]
# #### Inspect the train data
# Let's take a look at a few loaded samples:

# %%
import matplotlib.pyplot as plt

# Show some examples
fig, axs = plt.subplots(4, 4, figsize=(8, 8))

# Load the first 16 images and labels
xs = [train_mnist[i][0] for i in range(16)] # images
ys = [train_mnist[i][1] for i in range(16)] # labels

for i, ax in enumerate(axs.flatten()):
    x = xs[i]
    y = ys[i]
    x = x.permute((1, 2, 0))  # make channels last
    im = ax.imshow(x,  cmap = "gray")
    ax.set_title(f"Class {y}")
    ax.axis("off")

fig.colorbar(im, ax=axs, orientation='vertical', label="gray value", shrink = 0.9)
# %% [markdown]
# #### Dataloaders
# Now, from the loaded datasets (both the train and test splits), we derive the dataloaders. We use dataloaders as they provide additional load-time features.
# Specifically, a dataloader enables **iterating** over the dataset in batches. It provides **shuffling** if desired.
# Here, we set the `batch_size` for both the train and test loader, and set `shuffle` for training only.

# %%
from torch.utils.data import DataLoader

batch_size = 8
train_loader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

# %% [markdown]
# The dataset iterator and the dataloader iterator differ in shape:

# %%
# Dataset iterator
smpl, lbl = next(iter(train_mnist))
print(f"dataset element shape: {smpl.shape} (class: {lbl})")

# Dataloader iterator
smpl, lbl = next(iter(train_loader))
print(f"dataloader element shape: {smpl.shape} (class: {lbl})")

# %% [markdown]
# Dataloader elements come in **8** at a time, which is the `batch_size` we set above. Hence, the first tensor dimension is the "batch" dimension and has size **8**.
# The labels are in a tensor of size **8** as opposed to a single value like in the dataset case.
# Note that, both in the dataset and in the dataloader, the data is not presented as a 2d `28x28` image, but rather as `1x28x28` 3d piece of data.
# This is useful when using multichannel data, but in our case, this extra dimension is superfluous. We will therefore drop the channel dimension for training.
#
# In Summary:
# | | Dataset | DataLoader |
# |---|---|---|
# | Image shape | `(1, 28, 28)` | `(8, 1, 28, 28)` |
# | Dimensions  | Channel, Height (Y), Width (X) | Batch, Channel, Height (Y), Width (X) |
# | Label | scalar `int` | `tensor` of `batch_size` (8) `int` |
# | Batch dimension | ❌ | ✅ (size = `batch_size`) |

# %% [markdown]
# <div class="alert alert-info">
#     <b>Note:</b> the term <em>dimension</em> is used throughout this exercise in roughly 2 ways. One is engineering-centric and pertains to the structure of tensor objects, specifying how the data is laid out (<em>e.g. dataloader iterator, 4-dimensional tensors with a batch dimension, a channel dimension, a height dimension and a width dimension</em>). The other use refers to the mathematical spaces that the data exists in (<em>e.g. 2-d images of 28x28=784 pixels are viewed as vectors in 784 dimensions, latent vectors from a latent space may have only 256 feature dimensions</em>).<br>
#     Mind the context to avoid miss understandings.
# </div>

# %% [markdown]
# ## Part A.2: Variational Autoencoders (VAE)
# A Variational Autoencoder (VAE) is a machine learning architecture capable of learning a compressed representation of data by pushing it through a low-dimensional "bottleneck" and then expanding it back into its original size.
# The model is forced to rebuild with limited information, and must therefore learn to capture only the most important features, performing non-linear dimensionality reduction.
# <p align="center"><img src="assets/vae.png" width="100%"></p>
# In our case, we convert `28 * 28 = 784` pixel images into a **few** core features via the encoder part of the model. The decoder part then turns these few features back into `28 * 28` pixel images.
#
# ### Part A.2.1: The architecture of our VAE
# #### A shared backbone for encoder and decoder
# We choose a **Multilayer Perceptron (MLP)** as the architecture to back both the **encoder** and the **decoder**.
# MLPs consist of linear transformations (weights and biases) followed by non-linear activation functions (ReLU).

# %%
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim
                , hidden_dims=[], activation=nn.ReLU(), final_activation=None ):

        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation is not None: layers.append(activation)

        layers.append(nn.Linear(dims[-1], output_dim))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
# %% [markdown]
# #### Building the `VariationalAutoEncoder` class
#
# Here we provide a `VariationalAutoEncoder` class that uses the previously defined **MLP** class for the **encoder** and **decoder**.
#
# **The encode function**:
# * Reshapes the input by dropping the unused channel dimension.
# * Reduces width and height (28 x 28) to a single dimension (28 * 28 = 784).
# * Outputs `mu` and `logvar`; single-dimensional tensors that represent the center and logvariance for a given input in the latent space.
#
# **The reparametrization trick**:
# * Allows to sample the latent variable `z` as it if came from a Normal distribution with `mu` and $std = e^{logvar/2}$
# * However, sampling `z` directly from `mu` and $std$(`logvar`) does not allow for backpropagation (random sampling is not differentiable)
# * To allow for backpropagation, we isolate the non-differentiable random sampling node and sample $\epsilon$ from a Normal distritubion with mean 0 and standard deviation 1
# * We then use this $\epsilon$ to produce `z`: `z` $=$ `mu` $+ \epsilon * e^{logvar/2}$ (here, gradient can flow through `mu` and `logvar`)
# 
# ![Reparameterization trick](./assets/Reparameterization_Trick.png)
# Source: [Wikipedia](https://en.wikipedia.org/wiki/Reparameterization_trick#).
#
#
# **The decode function**:
# * Takes in a latent vector `z` and uses an MLP to reconstruct the original sample, reshaping as appropriate.
# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Task – Fill in the gaps</h2>
#
# There are gaps marked as `...`.
# Fill them in:
#
# **Missing in `def __init__`**
#
# The encoder and decoder instance of the MLP are missing. Replace `...` with:
# `self.encoder`
# `self.decoder`.
#
# How can you tell which is which?
#
#
# **Missing in `def reparameterize`**
#
# `epsilon` (missing twice)
# `std`
#
# Tip:
# $std = e^{logvar/2}$
# $z = mu + \epsilon * std$
#
#
# **Missing in `def forward`**
# `mu`
# `logvar`
# `z`
# `self.decode`
# `self.encode`
# `self.reparameterize`
# `xx` (this is the reconstructed image)
#
# </div>
#
#

# %% tags=["task"]
import torch

class VariationalAutoEncoder(nn.Module):
    def __init__( self, w, h, latent_dim
                , enc_hidden_dims=[256, 128, 64], enc_activation=nn.ReLU()
                , dec_hidden_dims=[64, 128, 256], dec_activation=nn.ReLU()
                ):
        super().__init__()
        self.w = w
        self.h = h
        data_dim = w * h

        # TODO
        ... = MLP( data_dim, latent_dim * 2 # latent_dim * 2, because it will be split into mu and logvar
                          , hidden_dims=enc_hidden_dims
                          , activation=enc_activation)

        # TODO
        ... = MLP( latent_dim, data_dim
                          , hidden_dims=dec_hidden_dims
                          , activation=dec_activation
                          , final_activation=nn.Sigmoid())






    def encode(self, x):
        b  = x.shape[0] # batchsize
        x_flat = x.reshape(b, -1) # (8, 1, 28, 28) → (8, 784)
        out = self.encoder(x_flat)
        mu, logvar = torch.chunk(out, 2, dim = 1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        ... = torch.exp(logvar / 2) # TODO
        ... = torch.randn_like(std) # TODO
        return ... * std + mu # TODO

    def decode(self, z):
        out = self.decoder(z)
        xx = out.reshape(-1, 1, self.w, self.h) # (8, 784) → (8, 1, 28, 28)
        return xx

    def forward(self, x):
        ..., ... = ...(x) # TODO
        ... = ...(mu, logvar)     # TODO
        ... = ...(z)           # TODO
        return xx, z, mu, logvar

# %% tags=["solution"]
import torch

class VariationalAutoEncoder(nn.Module):
    def __init__( self, w, h, latent_dim
                , enc_hidden_dims=[256, 128, 64], enc_activation=nn.ReLU()
                , dec_hidden_dims=[64, 128, 256], dec_activation=nn.ReLU()
                ):
        super().__init__()
        self.w = w
        self.h = h
        data_dim = w * h

        self.encoder = MLP( data_dim, latent_dim * 2 # latent_dim * 2, because it will be split into mu and logvar
                          , hidden_dims=enc_hidden_dims
                          , activation=enc_activation)
        self.decoder = MLP( latent_dim, data_dim
                          , hidden_dims=dec_hidden_dims
                          , activation=dec_activation
                          , final_activation=nn.Sigmoid())


    def encode(self, x):
        b  = x.shape[0] # batchsize
        x_flat = x.reshape(b, -1) # (8, 1, 28, 28) → (8, 784)
        out = self.encoder(x_flat)
        mu, logvar = torch.chunk(out, 2, dim = 1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mu

    def decode(self, z):
        out = self.decoder(z)
        xx = out.reshape(-1, 1, self.w, self.h) # (8, 784) → (8, 1, 28, 28)
        return xx

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xx = self.decode(z)
        return xx, z, mu, logvar
# %% [markdown]
# Run the following tests to confirm:

# %%
import inspect
import re

def normalize(s):
    return re.sub(r'\s+', '', s)

def test_vae(w=28, h=28, latent_dim=16, batch_size=8):
    vae = VariationalAutoEncoder(w, h, latent_dim)
    data_dim = w * h
    x = torch.randn(batch_size, 1, w, h)

    #  __init__: self.encoder
    assert hasattr(vae, 'encoder'), \
        "❌ 'self.encoder' missing or misspelled — correct it in __init__"
    try:
        enc_out = vae.encoder(torch.randn(batch_size, data_dim))
    except RuntimeError:
        raise AssertionError(
            f"❌ self.encoder crashed on input shape ({batch_size}, {data_dim}) "
            f"— did you swap self.encoder and self.decoder?"
        )
    assert enc_out.shape == (batch_size, latent_dim * 2), \
        f"❌ encoder output {enc_out.shape} ≠ ({batch_size}, {latent_dim*2}) — did you swap encoder/decoder?"
    print("✅ self.encoder: correct")

    #  __init__: self.decoder
    assert hasattr(vae, 'decoder'), \
        "❌ 'self.decoder' missing or misspelled — correct it in __init__"
    try:
        dec_out = vae.decoder(torch.randn(batch_size, latent_dim))
    except RuntimeError:
        raise AssertionError(
            f"❌ self.decoder crashed on input shape ({batch_size}, {latent_dim}) "
            f"— did you swap self.encoder and self.decoder?"
        )
    assert dec_out.shape == (batch_size, data_dim), \
        f"❌ decoder output {dec_out.shape} ≠ ({batch_size}, {data_dim}) — did you swap encoder/decoder?"
    print("✅ self.decoder: correct")

    #  reparameterize: check variable names
    norm_reparam = normalize(inspect.getsource(VariationalAutoEncoder.reparameterize))

    assert normalize('std = torch.exp(logvar / 2)') in norm_reparam, \
        "❌ reparameterize: first blank wrong"
    print("✅ reparameterize: 'std' correct")

    assert normalize('epsilon = torch.randn_like(std)') in norm_reparam, \
        "❌ reparameterize: second blank wrong"
    print("✅ reparameterize: 'epsilon' correct")

    assert  normalize('return epsilon * std + mu') in norm_reparam or \
            normalize('return std * epsilon + mu') in norm_reparam or \
            normalize('return mu + epsilon * std') in norm_reparam or \
            normalize('return mu + std * epsilon') in norm_reparam, \
            "❌ reparameterize: third blank wrong"
    print("✅ reparameterize: return statement correct")

    # forward: check variable names
    xx, z, mu, logvar = vae(x)
    assert xx.shape == (batch_size, 1, w, h), "❌ forward failed at runtime"
    norm_forward = normalize(inspect.getsource(VariationalAutoEncoder.forward))

    assert normalize('mu, logvar = self.encode(x)')         in norm_forward, \
        "❌ forward: check line 1"
    assert normalize('z = self.reparameterize(mu, logvar)') in norm_forward, \
        "❌ forward: check line 2"
    assert normalize('xx = self.decode(z)')                 in norm_forward, \
        "❌ forward: check line 3"
    print("✅ forward: all blanks correct")

    print("\n All tests passed!")

test_vae()

# %% [markdown]
# ### Part A.2.2: The loss functions
# Training a VAE balances two competing objectives:
#
# #### Reconstruction
# The reconstruction loss measures how well the decoder reconstructs an input image from the latent space.
# The **Binary Cross Entropy** reconstruction loss (**BCE loss**) is appropriate here, as we scale input images to gray values of [0, 1].
# The decoder's final activation is a Sigmoid function, mapping to the same scale of [0, 1].
#
#
# #### Latent space regularization
# **Kullback-Leibler divergence** loss (**KL loss**) measures how much a learned distribution of images in the latent space diverges from a standard normal distribution, i.e. a Normal distribution with mean 0 and std 1.
# KL-loss penalizes the latent space distribution for being different from a standard normal.

# %%
# The reconstruction loss
_bce_per_pixel = nn.BCELoss(reduction="none")
def rec_loss(xx, x):
    # sum over pixel dims, mean over batch — matches the kl_loss reduction
    per_image = _bce_per_pixel(xx, x).flatten(start_dim=1).sum(dim=1)
    return per_image.mean()

# %% [markdown]
#

# The KL loss
# %% tags=["task"]
def kl_loss(mu, logvar):
    return ...
    
# %% tags=["solution"]
def kl_loss(mu, logvar):
    # sum over latent dimensions, mean over batch
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))


# %% [markdown]
# #### Combined loss with beta weighting
# We combine the reconstruction loss and KL loss into an overall loss. The overall loss is what we backpropagate on.
# Parameter `beta` adds a weighting to the KL loss.
# `beta = 0` means only the reconstruction-loss influences training.

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Task – Overall loss</h2>
# Write code that weights the kl loss by beta and adds it to the reconstruction loss
# </div>

# %% tags=["task"]
def loss(rec, kl, beta):
    return ...


# %% tags=["solution"]
def loss(rec, kl, beta):
    return rec + beta * kl


# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Refreshed our acquaintance with MNIST
# - Explained how the dataset and dataloaders are used to iterate through large amounts of data in a structured manner
# - Familiarised ourselves with the MLP
# - Familiarised ourselves with the VAE model we are going to train
# - Prepared a composite loss, with a reconstruction term and a KLD term
#
# Next we will train our model.
# </div>

# %% [markdown]
# ### Part A.2.3: Training infrastructure
# Now we get to create and train our model on the MNIST dataset.
#
# #### A.2.3.1 Set the device
# As our model and dataset is small, CPUs are likely to outperform GPUs.
# The overhead of transferring the data to GPU might make the model slower than running it on CPU.
# %%
device = torch.device("cpu")

# %% [markdown]
# #### Part A.2.3.2: Model instance and optimizer
# **Model instance**
# We first create an instance of our `VariationalAutoEncoder` class. On construction, it needs to know the size of the data it will receive and the desired latent space size.
# We grab a sample from the dataset to derive the appropriate input size.
# We set the latent dimensions to **2**. A word of warning on this:

# %% [markdown]
# <div class="alert alert-info">
# <b> Attention: </b><code>latent_dim = 2 </code> is likely underparametrizing the latent space. We will lose a lot of relevant information when compressing a 28 x 28 image down to 2 dimensions. <br>
# However, we will roll with this for now as it helps us build an intution for the latent space.  <br>
# Later, we will retrain with a higher-dimensional latent space.
# </div>

# %%
data_sample, _ = next(iter(train_mnist))
_, w, h = data_sample.shape
model = VariationalAutoEncoder(w, h, latent_dim=2).to(device)

# %% [markdown]
# **Optimizer**
# We then create an optimizer for the model's parameters. During training, gradients will be computed from the backpropagation pass.
# The optimizer will read those gradients and update the model's parameters.

# %%
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=0.0001)

# %% [markdown]
# #### Part A.2.3.3: The training "loop"
# To train a model, the general idea is to iterate through the dataset, passing each element through the model to produce a reconstruction and embed it into the latent space.
# We observe how close to the original data the reconstruction is using the reconstruction loss function, and use that observation to inform the model optimisation.
# We penalize the latent space for not following a standard normal distribution using the KL-loss.
# Performing these steps going once through all the training data is what is referred to as a training **epoch**.
# We then loop this process over for a desired arbitrary number of training epochs.
#
# Below are three functions:
# `train_epoch`: Captures training for a single epoch and returns the average epoch loss.
# `train_epochs`: Calls `train_epoch` for the desired number of epochs and returns the losses per epoch.
# `plot_losses_live`: Plots losses during training and live-updates every few epochs.
# %%
from tqdm.auto import tqdm
from itertools import islice
from IPython.display import clear_output
import matplotlib.pyplot as plt


def train_epoch(model, loader, optimizer, loss, beta):
    model.train()
    running_rec_loss = 0.0
    running_kl_loss = 0.0
    running_loss = 0.0

    for x, _ in loader:
        x = x.to(device)
        xx, _, mu, logvar = model(x)
        rec_l = rec_loss(xx, x)
        kl_l = kl_loss(mu, logvar)
        l = loss(rec_l, kl_l, beta = beta)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        # Stats
        running_rec_loss += rec_l.item()
        running_kl_loss += kl_l.item()
        running_loss += l.item()

    avg_rec_loss = running_rec_loss / len(loader)
    avg_kl_loss = running_kl_loss / len(loader)
    avg_loss = running_loss / len(loader)

    return avg_rec_loss, avg_kl_loss, avg_loss



def plot_losses_live(epoch_losses, epoch_rec_losses, epoch_kl_losses):
    _, axes = plt.subplots(1, 3, figsize=(15, 3))

    for ax, values, title in zip(
        axes,
        [epoch_losses, epoch_rec_losses, epoch_kl_losses],
        ["Total Loss", "Reconstruction Loss", "KL Loss"]
    ):
        ax.plot(values, c = "k")
        ax.set_title(f"{title}: {values[-1]:.4f}")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def train_epochs(n, model, loader, optimizer, loss, beta, plot_every=10):
    epoch_rec_losses = []
    epoch_kl_losses = []
    epoch_losses = []

    for epoch in range(n):
        fresh_loader_iter = iter(loader)
        sliced_loader = tqdm(islice(fresh_loader_iter, 100), total=100, disable=True)
        avg_rec_loss, avg_kl_loss, avg_loss = train_epoch(
            model, sliced_loader, optimizer, loss, beta=beta
        )
        epoch_rec_losses.append(avg_rec_loss)
        epoch_kl_losses.append(avg_kl_loss)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % plot_every == 0:
            clear_output(wait=True)
            print(f"Epoch {epoch+1}/{n} | loss={avg_loss:.4f} | rec={avg_rec_loss:.4f} | kl={avg_kl_loss:.4f}")
            plot_losses_live(epoch_losses, epoch_rec_losses, epoch_kl_losses)

    return {"loss": epoch_losses, "rec": epoch_rec_losses, "kl": epoch_kl_losses}


# %% [markdown]
# ### Part A.2.4: Train a model for 100 epochs

# %% [markdown]
# We can now train `model`.
# In this part, we want to understand the model rather than analyze its results.
# Therefore, we sacrifice quality for time by only training 100 epochs.

# %%
epochs = 100
train_epochs(epochs, model, train_loader, optimizer, loss, beta = 1)

# %% [markdown]
# ### Part A.2.5: Inspect the trained model
# #### Get a test image
# Let's inspect what each part of the model does.
# We will probe the model with **one** image.
# Let's get it from the `test_loader`:

# %%
x, _ = test_loader.dataset[5] # reminder: test_loader is the dataloader for the test data

print(f"Image shape : {x.shape} --- C, Y, X") # Just one image!

plt.figure(figsize=(1,1))
plt.imshow(x.squeeze(), cmap="Grays");

# %% [markdown]
# #### The encoder
# Next, we pass the loaded image to the encoder, which returns `mu` and `logvar`.
# We previously set the latent space to two dimensions. Therefore, each `mu` and `logvar` have two entries.
#
# Let's have a look at `mu` and `logvar`

# %%
model.eval()
with torch.no_grad():
    mu, logvar = model.encode(x.unsqueeze(0).to(device))


print(f"mu    : {mu}. Dimensions: {len(mu[0])}")
print(f"logvar: {logvar}. Dimensions: {len(logvar[0])}")

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task: Compare size of an input image to the latent space</h3>
# The function <code>.nbytes</code> allows you to see the size of an array or tensor in bytes.  <br>
# Print the size of <code>x</code>.  <br>
# Compute the size of <code>mu</code> and the size of <code>logvar</code>. Print their sum.  <br>
# Which is bigger: the input image, or its latent space embedding?

# %% tags=["task"]
#Example:
example_tensor = torch.zeros(1)
print(example_tensor.nbytes)

# TODO:
...

# %% tags=["solution"]
example_tensor = torch.zeros(1)
print(example_tensor.nbytes)

print(f"Input image: {x.nbytes} bytes")
print(f"mu + logvar: {mu.nbytes + logvar.nbytes} bytes")

# %% [markdown]
# #### Sample
# Values `z` are sampled from a Normal distribution with mean `mu` and standard deviation $\sigma = e^{\,\text{logvar}/2}$
# The `reparametrize` function allows for this sampling without blocking backpropagation.
# Note how `z` is different for each draw. Here, we draw twice and call the result `z_1` and `z_2`.

# %%
with torch.no_grad():
    z_1 = model.reparameterize(mu, logvar)
    z_2 = model.reparameterize(mu, logvar) # same mu, same logvar

print(f"z_1: {z_1}, Dimensions: {len(z_1[0])}")
print(f"z_2: {z_2}, Dimensions: {len(z_2[0])}")
print("Are the two samples the same?", z_1 == z_2 )


# %% [markdown]
# #### The decoder
# Next, we can decode `z_1` and `z_2`.
# The decoder reconstructs image `rec_1` from `z_1`, and image `rec_2` from `z_2`

# %%

with torch.no_grad():
    rec_1 = model.decode(z_1).cpu().numpy().squeeze()
    rec_2 = model.decode(z_2).cpu().numpy().squeeze()

# %% [markdown]
# Are the two reconstructed images the same?

# %%
fig, ax = plt.subplots(1, 4)

im0 = ax[0].imshow(x.squeeze(), cmap= "Grays")
ax[0].set_title("original")
plt.colorbar(im0, shrink = 0.2)

im0 = ax[1].imshow(rec_1, cmap= "Grays")
ax[1].set_title("rec 1")
plt.colorbar(im0, shrink = 0.2)

im1 = ax[2].imshow(rec_2, cmap= "Grays")
ax[2].set_title("rec 2")
plt.colorbar(im1, shrink = 0.2)

im2 = ax[3].imshow(rec_1 - rec_2, cmap= "RdBu")
ax[3].set_title("rec 1 - rec 2")
plt.colorbar(im2, shrink = 0.2)

plt.tight_layout()


# %% [markdown]
# Clearly not! VAEs are *probabilistic* models.

# %% [markdown]
# #### Display multiple reconstructions
# Now let's look at what reconstructions look like for multiple images.
# Below are a couple simple visualisation function to display original and reconstructed images together, and to query the model for a batch of reconstructions and display them using the first function.

# %%
def show_recon(og, recon):
    """
    og and recon: Tensors of shape (B, 1, 28, 28) or (B, 784)
    """
    b, c, h, w = og.shape
    og = og.view(-1, h, w).detach().cpu()
    recon = recon.view(-1, h, w).detach().cpu()

    fig, axes = plt.subplots(nrows=2, ncols=b, figsize=(b * 1.5, 3))

    for i in range(b):
        # Top row: Original
        axes[0, i].imshow(og[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")

        # Bottom row: Reconstruction
        axes[1, i].imshow(recon[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Recon")

    plt.show()

#import torch
def view_test_sample(model, loader):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images.to(device)
        recon, _, _, _ = model(images)
    show_recon(images, recon)


# %% [markdown]
# Let's call this on our model as it currently stands.

# %%
view_test_sample(model, test_loader)

# %% [markdown]
# Not great, not terrible... After the checkpoint, we will instantiate new models and train them for longer.

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Instanciated our model and an optimizer
# - Provided a training loop which goes through the data samples, run the model, and steps the optimizer to update the model's parameters
# - Trained the model for a few "epochs" through all the training samples and observed the loss values
# - Inspected mu and logvar produced by the encoder
# - Inspected the reconstructions produced by the decoder
#
# Next we will train our model for a larger number of epochs.
# </div>

# %% [markdown]
# ### Part A.2.6: Train two models for 1000 epochs

# %% [markdown]
# #### A.2.6.1: Train a model without regularized latent sapce

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Task</h2>
# We will now train two models, `model0` without regularization and `model1` with regularization.
# To acheive this, we set the `beta` parameter for the loss used with `model0` to `0`.
#
# Let's train our first "serious" model.
# * Instantiate a new variational autoencoder model and name it `model0`
# * Keep `latent_dim = 2`. This is not ideal, but helps us better understand the latent space.
# * Instantiate a new optimizer
# * Pass `beta = 0`
# * Train your new model for `epochs = 1000`
#
#
#
# </div>
#

# %% tags=["task"]

model0 = VariationalAutoEncoder(...).to(device)  # TODO
optimizer = Adam(model0.parameters(), lr=0.0001)         # fresh optimizer
...
...
losses0 = train_epochs(epochs, model0, train_loader, optimizer, loss, beta = beta)

# %% tags=["solution"]

model0 = VariationalAutoEncoder(w, h, latent_dim = 2).to(device)  # fresh weights
optimizer = Adam(model0.parameters(), lr=0.0001)         # fresh optimizer

epochs = 1000
beta = 0
losses0 = train_epochs(epochs, model0, train_loader, optimizer, loss, beta = beta)

# %% [markdown]
# Let's have a look at the results. Reconstructed images should now look better.

# %%
view_test_sample(model0, test_loader)

# %% [markdown]
# <div class="alert alert-block alert-warning"><h4> Questions </h4>
#
# * Which loss is decreasing more? Why?
#
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Trained the model for a large number of epochs
# - Observed several reconstructions, comparing we previously obtained results
#
# Next we will attempt to regularize the model's latent space.
# </div>

# %% [markdown]
# #### A.2.6.2: Train a model with regularized latent sapce
# We are now training another model called `model1`

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Task</h2>
#
# Change one variable in the code below to train a new model with better-behaved KL loss.
# In this case, the KL loss doesn't have to decrease – we just want it to increase less.
#
# Tips:
# * Have a look at the overall loss function definitions
# * Look at the order of magnitude of the reconstruction loss and KL loss, for instance at epoch 1000, to decide on a value
# * You can train for fewer epochs if you want to try multiple values. Train for 1000 epochs once you decided
# </div>

# %% tags=["task"]
model1 = VariationalAutoEncoder(w, h, latent_dim=2).to(device)
optimizer = Adam(model1.parameters(), lr=0.0001)         # fresh optimizer
epochs = 1000
beta = # TODO
losses1 = train_epochs(epochs, model1, train_loader, optimizer, loss, beta = beta)


# %% tags=["solution"]
# beta 1
model1 = VariationalAutoEncoder(w, h, latent_dim=2).to(device)
optimizer = Adam(model1.parameters(), lr=0.0001)         # fresh optimizer
epochs = 1000
beta = 1
losses1 = train_epochs(epochs, model1, train_loader, optimizer, loss, beta = beta)


# %% [markdown]
# Let's have a look at reconstructions for model1:

# %%
view_test_sample(model1, test_loader)


# %% [markdown]
# Let's compare the losses:

# %%
def plot_losses_compare(loss_dicts, labels, colors=None):
    """
    loss_dicts: list of dicts, e.g. [losses_1, losses_2]
    labels:     list of strings, e.g. ["Beta 0.001", "Beta 1"]
    colors:     optional list of colors, e.g. ["blue", "red"]
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default matplotlib colors

    keys = list(loss_dicts[0].keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 3))

    for i, key in enumerate(keys):
        for loss_dict, label, color in zip(loss_dicts, labels, colors):
            axes[i].plot(loss_dict[key], label=label, color=color, marker='o', markersize=2)
        axes[i].set_title(key)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()

    plt.tight_layout()
    plt.show()


plot_losses_compare(
    loss_dicts=[losses0, losses1],
    labels=["Beta 0", "Beta high"],
)


# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Trained another model on our dataset
# - Observed how changing the beta parameter in the composite loss (the factor in front of the KLD term) affects the overall loss
#
# Next we will use our trained model to explore a set of test images (not yet seen by the model) and explore obtained results.
# </div>

# %% [markdown]
# ### Part A.2.7: Apply the models to test images
# #### Part A.2.7.1: Get latent space properties

# %% [markdown]
# Previously, we encoded a few images into the latent space. Now we will encode the the entire test set.
# Below is a function that receives a model and a dataloader.
#
# The function returns:
# | Return value | Shape | Description |
# |---|---|---|
# | `mus` | `(10000, 2) (images, dims)` | latent mean for every test image |
# | `logvars` | `(10000, 2) (images, dims)` | latent log-variance for every test image |
# | `lbls` | `(10000,) (images,)` | digit label for every test image |
# | `mu_mean` | `(10, 2) (digit class, dims)` | average latent position per digit class |

# %%
def get_latent_features(model, loader):
    model.eval()
    latents = []
    logvars = []
    labels = []

    with torch.no_grad():
        for x, lbl in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            latents.append(mu.cpu())
            logvars.append(logvar.cpu())
            labels.append(lbl)

    mus = torch.cat(latents, dim=0)
    logvars = torch.cat(logvars, dim=0)
    lbls = torch.cat(labels, dim=0)

    mu_mean = torch.stack([mus[lbls == i].mean(dim=0) for i in range(10)])
    return mus, logvars, lbls, mu_mean



# %% [markdown]
# Let's get these features for `model0` and `model1`:

# %%

#get all latent features
# model 0
mus_model0, logvars0, lbls0, mu_mean0 = get_latent_features(model0, tqdm(test_loader))

# model 1
mus_model1, logvars1, lbls1, mu_mean1 = get_latent_features(model1, tqdm(test_loader))


print(f"mu shape: {mus_model0.shape}, labels shape: {lbls0.shape}")

# %% [markdown]
# #### Part A.2.7.2: Visualize the latent spaces.
# **Let's plot the two dimensions of the latent space**.
# Every data-point represents one test-image. We plot the two latent dimensions (mu₁, mu₂) for `model0` and `model1`

# %%
import numpy as np

def scatter_digits(ax, mus, lbls, mu_mean=None, alpha=1, CMAP = "tab10"):
    """Colour-coded scatter, one series per digit so legend works."""

    CMAP = plt.get_cmap(CMAP)

    for d in range(10):
        mask = lbls == d
        ax.scatter(mus[mask, 0], mus[mask, 1], s=1, color=CMAP(d),
                   label=str(d), rasterized=True, alpha=alpha)

    if mu_mean is not None:
        for d in range(10):
            ax.scatter(*mu_mean[d], s=100, color="white", edgecolors="white", linewidths=0.4, marker="X", zorder=9)
            ax.scatter(*mu_mean[d], s=70,  color=CMAP(d), edgecolors="black", linewidths=0.5, marker="X", zorder=10)

    ax.legend(title="Digit", markerscale=6, ncol=2, fontsize=7, loc="best")
    ax.set_aspect("equal")


def scatter_with_normal(ax, mus, lbls, rnd_normal, mean, std):
    """Digits on top of magenta N(0,1) cloud."""
    ax.scatter(rnd_normal[:, 0], rnd_normal[:, 1],
               s=1, alpha=0.25, color="magenta", label=f"N({mean},{std})", rasterized=True, zorder=5)
    ax.scatter(mus[:, 0], mus[:, 1],
               s=1, color="gray", label="mu", rasterized=True, zorder=2)
    ax.legend(markerscale=6, ncol=2, fontsize=7, loc="best")
    ax.set_aspect("equal")


def plot_latent_digits(mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1, alpha=1, CMAP = "tab10"):
    """Latent space coloured by digit, with per-class centroids."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mus, lbls, mu_mean, title in [
        (axes[0], mus_model0, lbls0, mu_mean0, "model0: β=0 latent space"),
        (axes[1], mus_model1, lbls1, mu_mean1, "model1: β>0 latent space"),
    ]:
        scatter_digits(ax, mus, lbls, mu_mean=mu_mean, alpha=alpha, CMAP=CMAP)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("mu₁"); ax.set_ylabel("mu₂")
    fig.suptitle("Latent space — coloured by digit", fontsize=13)
    plt.tight_layout(); plt.show()




def plot_latent_vs_normal(mus_model0, lbls0, mus_model1, lbls1, rnd_normal,
                          clf0=None, clf1=None, mean=0, std=1, resolution=500):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mus, lbls, clf, title in [
        (axes[0], mus_model0, lbls0, clf0, f"β=0 vs N({mean},{std})"),
        (axes[1], mus_model1, lbls1, clf1, f"β>>0 vs N({mean},{std})"),
    ]:
        # Decision boundaries (if a classifier is provided )
        if clf is not None:
            x_min, x_max = mus[:, 0].min() - 0.5, mus[:, 0].max() + 0.5
            y_min, y_max = mus[:, 1].min() - 0.5, mus[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                 np.linspace(y_min, y_max, resolution))
            grid_preds = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            # ax.contourf(xx, yy, grid_preds, levels=np.arange(-0.5, 10, 1),
            #             cmap="tab10", alpha=0.25, zorder=0)
            ax.contour(xx, yy, grid_preds, levels=np.arange(-0.5, 10, 1),
                       colors="k", linewidths=0.4, zorder=1)

        # Normal prior cloud + latent points
        scatter_with_normal(ax, mus, lbls, rnd_normal, mean, std)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("mu₁"); ax.set_ylabel("mu₂")

    fig.suptitle(f"Latent space vs N({mean},{std}) prior", fontsize=13)
    plt.tight_layout()
    plt.show()




#call
plot_latent_digits(mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1)

mean = 0
std = 1
plot_latent_vs_normal(mus_model0, lbls0, mus_model1, lbls1, rnd_normal=np.random.normal(mean, std, size=(10000, 2)), mean = mean, std = std)


# %% [markdown]
# **top row**
# Each embedded image is color-coded by digit class. The marker x shows the per-class centroid – the average mu across all images of that digit.
#
# **bottom row**
# Encoded digits (gray) are overlayed with 10000 samples from a standard normal distribution N(0,1) prior to assess how well the learned latent distribution matches it.

# %% [markdown]
# <div class="alert alert-block alert-warning"><h4> Questions </h4>
# <ul>
# <li>Look at the per-class centroids for a model of your choice. Do you think there's a reason why some centroids are closer to each other, while others are more distant? </li>
# <li>Which model's latent space is more similar to a standard normal distribution? Why is this the case?</li>
# </ul>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Extracted 2-d latent features for test images for both our trained models
# - Visualised the extracted 2-d mean features and discussed classes distributions
#
# Next we will explore 2-d logvars.
# </div>

# %% [markdown]
# **visualizing `logvar`**
#
# We plotted `mu`, but what about `logvar`?
#
# So far we have only visualized `mu`, the center of each images distribution in latent space.
# The encoder also outputs `logvar`, which controls the spread of each images' distribution.
#
# Below we will sample 100 values `z` from `mu` and `logvar`.
#
# So each of the 10000 test-image is represented by 100 points – 1000000 points in total.

# %%
def sample_from_latents(mus, logvars, n_samples=100):
    """
    For each (mu, logvar) pair, draw n_samples via the reparameterization trick.
    Returns z_samples of shape (N * n_samples, 2).
    """
    N, latent_dim = mus.shape
    # expand to (N, n_samples, latent_dim)
    mu_exp     = mus.unsqueeze(1).expand(-1, n_samples, -1)
    logvar_exp = logvars.unsqueeze(1).expand(-1, n_samples, -1)

    std = torch.exp(logvar_exp / 2)
    eps = torch.randn_like(std)
    z   = mu_exp + eps * std                        # (N, n_samples, 2)
    return z.reshape(N * n_samples, latent_dim)     # (N*n_samples, 2)


# ── sample ─────────────────────────────────────────────────────────────────
n_samples = 100

with torch.no_grad():
    z_samples0 = sample_from_latents(mus_model0, logvars0, n_samples)
    z_samples1 = sample_from_latents(mus_model1, logvars1, n_samples)

# repeat each label n_samples times to match the expanded z array
lbls0_rep = lbls0.repeat_interleave(n_samples)
lbls1_rep = lbls1.repeat_interleave(n_samples)

# ── plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 7))

ax = axes[0]
scatter_digits(ax, z_samples0.numpy(), lbls0_rep, mu_mean=mu_mean0, alpha = 0.1)
ax.set_title("β=0 — 100 samples per posterior", fontsize=11)
ax.set_xlabel("z₁"); ax.set_ylabel("z₂")

ax = axes[1]
scatter_digits(ax, z_samples1.numpy(), lbls1_rep, mu_mean=mu_mean1, alpha = 0.1)
ax.set_title("β=high — 100 samples per posterior", fontsize=11)
ax.set_xlabel("z₁"); ax.set_ylabel("z₂")

plt.suptitle("Sampled z", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-warning"><h4> Questions </h4>
# <ul>
# <li>Which latent space looks more continuous? </li>
# <li>Why is it important that the latent space follows a known distribution?</li>
# </ul>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Visualised extracted 2-d logvars
# - Discussed how beta affects the latent space
#
# Next we will explore classification using extracted latent features.
# </div>

# %% [markdown]
# #### Part A.2.7.3: Classification in the latent space
# The previous plots show clusters of numbers emerging. But can we quantify how well these clusters are separated?
#
# Next, we train a **logistic regression classifier** on the latent space.
# This is asking how well a **simple linear model** can distinguish digits
# using only their latent coordinates (μ₁, μ₂) as features.
#
# If the clusters are truly well-separated, logistic regression — which can
# only draw straight decision boundaries — should achieve high accuracy.
#
# Note we're compressing the entire MNIST image (784 pixels) down to just
# **2 numbers**, so the accuracy also reflects how much class-relevant
# information is preserved in the bottleneck.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def logreg(mus, lbls, test_size=0.2, random_state=42):

    mus_train, mus_val, lbls_train, lbls_val = train_test_split(
        mus.numpy(), lbls.numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=lbls.numpy()  # ensures each split has the same class proportions
    )

    # Fit only on training data
    clf = LogisticRegression(max_iter=1000)
    clf.fit(mus_train, lbls_train)

    # Evaluate only on validation data
    preds_val = clf.predict(mus_val)
    accuracy = accuracy_score(lbls_val, preds_val)

    unique_lbls = sorted(set(lbls_val))
    conf_m = confusion_matrix(lbls_val, preds_val, labels=unique_lbls)

    return accuracy, unique_lbls, conf_m, clf

accuracy0, unique_lbls0, conf_m0, clf0 = logreg(mus_model0, lbls0)
accuracy1, unique_lbls1, conf_m1, clf1 = logreg(mus_model1, lbls1)


# %% [markdown]
# We can visualize the decision boundaries:

# %%
def plot_decision_boundaries(ax, clf, mus, lbls, title, resolution=500, CMAP="tab10"):
    """Shade decision regions of a fitted classifier over the 2-D latent space."""
    CMAP_obj = plt.get_cmap(CMAP)

    x_min, x_max = mus[:, 0].min() - 0.5, mus[:, 0].max() + 0.5
    y_min, y_max = mus[:, 1].min() - 0.5, mus[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_preds = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Shade decision regions (low alpha so data points stay visible)
    ax.contourf(xx, yy, grid_preds, levels=np.arange(-0.5, 10, 1),
                cmap=CMAP, alpha=0.25, zorder=0)

    # Draw crisp decision boundaries
    ax.contour(xx, yy, grid_preds, levels=np.arange(-0.5, 10, 1),
               colors="k", linewidths=0.4, zorder=10)

    # Overlay data points
    scatter_digits(ax, mus, lbls)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("mu₁"); ax.set_ylabel("mu₂")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_decision_boundaries(axes[0], clf0, mus_model0.numpy(), lbls0.numpy(), "β=0 — decision boundaries")
plot_decision_boundaries(axes[1], clf1, mus_model1.numpy(), lbls1.numpy(), "β>0 — decision boundaries")
fig.suptitle("Logistic-regression decision boundaries in latent space", fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# Now let's plot the confusion matrix:

# %%
def confmatrix(ax, conf_m, accuracy, unique_lbls, title=""):
    ax.imshow(conf_m)
    ax.set_title(f"{title}\nAccuracy: {accuracy:.4f}")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(conf_m.shape[0]):
        for j in range(conf_m.shape[1]):
            ax.text(j, i, conf_m[i, j],
                    ha="center", va="center", color="w")
    ax.set_xticks(unique_lbls)
    ax.set_yticks(unique_lbls)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

confmatrix(axes[0], conf_m0, accuracy0, unique_lbls0, title="β=0")
confmatrix(axes[1], conf_m1, accuracy1, unique_lbls1, title="β>>0")

plt.suptitle("Logistic Regression on Latent Space (μ₁, μ₂)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-warning"><h4> Questions </h4>
# <ul>
# * Question: is one classification better than the other?  <br>
# * If so, why do you think is the classification accuracy higher in one model than in the other?
# </ul>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Trained a logistic regression classifier using the extracted mean latent features
# - Presented how the classifier divides the space showing decision boundaries overlaid on top of our latent space visualisation
# - Presented a confusion matrix to establish the accuracy reached thanks to our latent space (contrasting results with and without KLD regularisation)
#
# Next we will explore sampling from the latent space directly.
# </div>

# %% [markdown]
# ### A.2.8. Sample from the latent space
# So far, we have looked at the latent space. Next, we look at what the decoder produces when given a latent vector.

# %% [markdown]
# Function `gen_mean_numbers` takes `mu_mean`, the average latent position of all test images of digit i.
# These correspond to the centroids shown in the plots above.
#
# The result is the most typical representation of each digit as embedded by the model.
#
# In a latent-space with well-separated clusters, each centroid should look like a clean version of the digit.

# %%
# Uncomment and run if you want to see the latent-space again:
plot_latent_digits(mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1)


# %%
# Plot average latent vectors

def gen_mean_numbers(model, mu_mean, title):
    fig, ax = plt.subplots(1, 10, figsize = (20, 4))
    with torch.no_grad():
        for i in range(10):
            gen = model.decode(mu_mean[i].to(device))

            ax[i].imshow(gen.detach().cpu().squeeze().numpy(), cmap = "Grays")
            ax[i].set_title(f"Mean for {i}")
    fig.suptitle(title)
    plt.tight_layout()


# gen_mean_numbers(model0, mu_mean0, title="Model 0") # model with beta = 0
gen_mean_numbers(model1, mu_mean1, title="Model 1") # model with beta >> 0

# %% [markdown]
# #### Interpolate between digits in the latent space
# Above, we have seen "mean" representations for each digit, corresponding to centroids of each digit class in the latent space.
# We can decode points sampled along the straight line between two centroids in latent space. A visualization will be plotted below.
#

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# Can we interpolate and reconstruct the same way with different architectures? (AE, ResNet, UNet, etc...)
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Task</h2>
# Run the code below without modifications. You should see reconstructions of interpolated images along the path between the centroids of 0 and 6. <br>
# Next, choose your own two digit centroids. Latent space permitting, try three types of paths:
#
# * Gap — the line passes through an empty region of the latent space   <br>
# * Traversal — the line passes through one or more other digit clusters<br>
# * Clean — the line connects two digits directly, with no gaps and no other clusters in between<br>
#
#
# For each case: can you explain what the decoded images look like, and why?
# </div>

# %%
# TODO: pick digits
digit_a = 0
digit_b = 6
steps = 12 # number of times to sample


def plot_latent_digits_interp(mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1, digit_a, digit_b):
    """Latent space coloured by digit, with a line connecting two class centroids."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, mus, lbls, mu_mean, title in [
        (axes[0], mus_model0, lbls0, mu_mean0, "β=0 latent space"),
        (axes[1], mus_model1, lbls1, mu_mean1, "β=high latent space"),]:
        scatter_digits(ax, mus, lbls, mu_mean=mu_mean)

        # ── line between the two class centroids ──────────────────────────
        pts = np.stack([mu_mean[digit_a], mu_mean[digit_b]])
        ax.plot(pts[:, 0], pts[:, 1], c="black", lw=2, ls="--", zorder=20)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("mu₁"); ax.set_ylabel("mu₂")

    fig.suptitle(f"Latent space — interpolation path {digit_a} → {digit_b}", fontsize=13)
    plt.tight_layout(); plt.show()




# interpolate between two digits
def interpolate(digit_a, digit_b, mu_mean, model, title, steps = 10):

    mu_a = mu_mean[digit_a]
    mu_b = mu_mean[digit_b]

    _, ax = plt.subplots(1, steps, figsize = (20, 3))

    with torch.no_grad():
        for i, weight_b in enumerate(np.linspace(0, 1, steps)):
            weight_a = 1 - weight_b
            trajectory = weight_a * mu_a + weight_b * mu_b
            trajectory = trajectory.to(device)
            gen = model.decode(trajectory)

            ax[i].imshow(gen.detach().cpu().squeeze().numpy(), cmap = "Grays")
            ax[i].set_title(f"{weight_a:.1f} x $\mu_{digit_a}$, {weight_b:.1f} x $\mu_{digit_b}$")
    plt.suptitle(title)
    plt.tight_layout()



plot_latent_digits_interp(mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1, digit_a=digit_a, digit_b=digit_b)

interpolate(digit_a, digit_b, mu_mean0, model0, "Beta = 0", steps = steps)
interpolate(digit_a, digit_b, mu_mean1, model1, "Beta > 0", steps = steps)

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Bonus task</h2>
# Change coordinates <code>z = [mu₁, mu₂]</code> to decode any point in the latent spaces.  <br>
# <code>z</code> will appear as a magenta star.
# </div>

# %%
# TODO: pick a point in the latent space
MU1 = -20
MU2 = -12
z = [MU1, MU2]

def decode_point(z, model0, model1, mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1):
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, 4, figsize=(12, 8))

    for ax_scatter, ax_img, model, mus, lbls, mu_mean, title in [
        (axes[0], axes[1], model0, mus_model0, lbls0, mu_mean0, "β=0"),
        (axes[2], axes[3], model1, mus_model1, lbls1, mu_mean1, "β>0"),
    ]:
        # latent space with marked point
        scatter_digits(ax_scatter, mus, lbls, mu_mean=mu_mean)
        ax_scatter.scatter(*z, s=200, color="magenta", marker="*", zorder=20)
        ax_scatter.set_title(f"{title} latent space")
        ax_scatter.set_xlabel("mu₁"); ax_scatter.set_ylabel("mu₂")

        # decoded image
        with torch.no_grad():
            gen = model.decode(z_tensor)
        ax_img.imshow(gen.cpu().squeeze(), cmap="gray")
        ax_img.set_title(f"{title} decoded")
        ax_img.axis("off")

    plt.tight_layout()
    plt.show()

decode_point(z, model0, model1, mus_model0, lbls0, mu_mean0, mus_model1, lbls1, mu_mean1)

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Sampled arbitrary points in the latent space (even points <b>not</b> corresponding to existing images)
# - Produced a reconstruction image for the sampled points
# - Discussed on the observed digits
#
# Next we will expand the dimensionality of our latent space.
# </div>

# %% [markdown]
# ## A.2.9: Higher-dimensional latent-spaces
#
# So far, we've only looked at a models with two latent dimensions.
# It is easier to visualize and intuitively understand two dimensions, but two numbers may not be enough to capture all meaningful variation of the dataset.
# Digits may overlap in latent-space, because two dimensions do not offer enough room to separate them out.
#
# Let's train a model with higher-dimensional latent sapce. Below, we
# * Instantiate a new variational autoencoder model and name it `model2`
# * Instantiate a new optimizer
# * Pass `beta = 1`
# * Train your new model for `epochs = 1000`

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>Task</h2>
#
# Choose `latent_dim = n`, where n is bigger than 5 and smaller than 100
#
# </div>
#

# %% tags=["task"]
latent_dim = ...

model2 = VariationalAutoEncoder(w, h, latent_dim = latent_dim).to(device)
optimizer = Adam(model2.parameters(), lr = 0.0001)

epochs = 1000
beta = 1

train_epochs(epochs, model2, train_loader, optimizer, loss, beta = beta);

# %% tags=["solution"]
latent_dim = 99

model2 = VariationalAutoEncoder(w, h, latent_dim = latent_dim).to(device)
optimizer = Adam(model2.parameters(), lr = 0.0001)

epochs = 1000
beta = 1

train_epochs(epochs, model2, train_loader, optimizer, loss, beta = beta);

# %% [markdown]
# Let's have a look at a few reconstructions:

# %%
view_test_sample(model2, test_loader)

# %% [markdown]
# Let's fast-forward – get latent features

# %%
mus2, logvars2, lbls2, mu_mean2 = get_latent_features(model2, tqdm(test_loader))

print(mus2.shape) # N samples, latent dims
print(logvars2.shape) # N samples, latent dims
print(lbls2.shape) # N samples
print(mu_mean2.shape) # N digits, latent dims


# %% [markdown]
# #### Dimensionality reduction on the latent space
# We now have too many latent dimensions to easily visualize.
#
# We therefore apply **UMAP** (Uniform Manifold Approximation and Porjection). UMAP is a dimensionality reduction technique that non-linearly projects the high-dimensional latent space to 2D.
# It attempts to keep points that are close in high-dimensions close in their 2D projection.
#
# `run_umap` performs this projection on `mus2` (the posterior mean in the latent space of all test images) and `mu_mean`, the per-class centroid in the latent space.
#
# Note that UMAP is a visualization tool. We can use it to build intuition, but should not draw conclusions about the geometry of the latent space.

# %%
from umap import UMAP

def run_umap(latents, n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, means=None):
    reducer = UMAP(n_components=n_components, random_state=random_state,
                   n_neighbors=n_neighbors, min_dist=min_dist)
    if means is not None:
        combined = np.concatenate([latents.numpy(), means.numpy()])
        combined_2d = reducer.fit_transform(combined)

        # Split back into latents and means
        mu_2d = combined_2d[:len(latents)]        # (10000, 2)
        mu_mean_2d = combined_2d[len(latents):]  # (10, 2)
        return mu_2d, mu_mean_2d

    mu_2d = reducer.fit_transform(latents.numpy())
    return mu_2d



# %%
mu_2d_2, mu_means_2d_2 = run_umap(mus2, means = mu_mean2)
_, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter_digits(ax, mu_2d_2, lbls2, mu_means_2d_2, alpha=0.6)

# %% [markdown]
# Clusters look well-separated, but we should verify with logistic regression.
# Below, we call the logistic regression function again and plot the confusion matrix.

# %%
accuracy2, unique_lbls2, conf_m2, clf2 = logreg(mus2, lbls2)


fig, axes = plt.subplots(1, 1, figsize=(5, 5))

confmatrix(axes, conf_m2, accuracy2, unique_lbls2, title= f"β=1, latent_dims = {latent_dim}")

plt.suptitle("Logistic Regression on Latent Space ", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint</h2>
# Let us know when you've reached this point!
#
# At this point we have:
#
# - Trained a new model with a (relatively) large amount of latent features
# - Discussed the options to visualise such a high-dimensional space, and provided a UMAP-based visualisation
#
# We have now finished PART A about representation learning. Take a break and return for PART B about explainable AI.
# </div>

# %% [markdown]
# # PART B: Explainable AI (XAI)
# ## Part B.1: Setup
#
# In this part of the notebook, we will use the colored MNIST dataset.
# We will also learn to load one of our trained classifiers from a checkpoint.

# %%
# loading the data
from classifier.data import ColoredMNIST

mnist = ColoredMNIST("extras/data", download=True)
# %% [markdown]
# Some information about the dataset:
# - The dataset is a colored version of the MNIST dataset.
# - Instead of using the digits as classes, we use the colors.
# - There are four classes - the goal of the exercise is to find out what these are.
#
# Let's plot some examples
# %%
import matplotlib.pyplot as plt

# Show some examples
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    x, y = mnist[i]
    x = x.permute((1, 2, 0))  # make channels last
    ax.imshow(x)
    ax.set_title(f"Class {y}")
    ax.axis("off")

# %% [markdown]
# During the setup you have pre-traiend a classifier on this dataset. It is the same architecture classifier as you used in the Failure Modes exercise: a `DenseModel`.
# Let's load that classifier now!
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 1.1: Load the classifier</h3>
# We have written a slightly more general version of the <code>DenseModel</code> that you used in the previous exercise. Ours requires two inputs:
# <li> <code>input_shape</code>: the shape of the input images, as a tuple </li>
# <li> <code>num_classes</code>: the number of classes in the dataset </li>
#
# Create a dense model with the right inputs and load the weights from the checkpoint.
# </div>
# %% tags=["task"]
import torch
from classifier.model import DenseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO Load the model with the correct input shape
model = DenseModel(input_shape=(...), num_classes=4)

# TODO modify this with the location of your classifier checkpoint
checkpoint = torch.load(...)
model.load_state_dict(checkpoint)
model = model.to(device)
# %% tags=["solution"]
import torch
from classifier.model import DenseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model
model = DenseModel(input_shape=(3, 28, 28), num_classes=4)
# Load the checkpoint
checkpoint = torch.load("extras/checkpoints/model.pth")
model.load_state_dict(checkpoint)
model = model.to(device)

# %% [markdown]
# Don't take my word for it! Let's see how well the classifier does on the test set.
# %%
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm  # This is a nice library for showing progress bars

test_mnist = ColoredMNIST("extras/data", download=True, train=False)
dataloader = DataLoader(test_mnist, batch_size=32, shuffle=False)

labels = []
predictions = []
for x, y in tqdm(dataloader):
    pred = model(x.to(device))
    labels.extend(y.cpu().numpy())
    predictions.extend(pred.argmax(dim=1).cpu().numpy())

cm = confusion_matrix(labels, predictions, normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()
# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 1</h2>
#
# At this point we have:
#
# - Loaded a classifier that classifies MNIST-like images by color, but we don't know how!
#
# We will not stop here as a group, it's just the end of Part 1. So continue on with part 2 right away.
# </div>
# %% [markdown]
# # Part 2: Using Integrated Gradients to find what the classifier knows
#
# In this section we will make a first attempt at highlighting differences between the "real" and "fake" images that are most important to change the decision of the classifier.
#

# %% [markdown]
# ## Attributions through integrated gradients
#
# Attribution is the process of finding out, based on the output of a neural network, which pixels in the input are (most) responsible for the output. Another way of thinking about it is: which pixels would need to change in order for the network's output to change.
#
# Here we will look at an example of an attribution method called [Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients). If you have a bit of time, have a look at this [super fun exploration of attribution methods](https://distill.pub/2020/attribution-baselines/), especially the explanations on Integrated Gradients.

# %% tags=[]
batch_size = 4
batch = []
for i in range(4):
    batch.append(next(image for image in mnist if image[1] == i))
x = torch.stack([b[0] for b in batch])
y = torch.tensor([b[1] for b in batch])
x = x.to(device)
y = y.to(device)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 2.1 Get an attribution</h3>
#
# In this next part, we will get attributions on single batch. We use a library called [captum](https://captum.ai), and focus on the `IntegratedGradients` method.
# Create an `IntegratedGradients` object and run attribution on `x,y` obtained above.
#
# </div>

# %% tags=["task"]
from captum.attr import IntegratedGradients

############### Task 2.1 TODO ############
# Create an integrated gradients object.
integrated_gradients = ...

# Generated attributions on integrated gradients
attributions = ...

# %% tags=["solution"]
#########################
# Solution for Task 2.1 #
#########################

from captum.attr import IntegratedGradients

# Create an integrated gradients object.
integrated_gradients = IntegratedGradients(model)

# Generated attributions on integrated gradients
attributions = integrated_gradients.attribute(x, target=y)

# %% tags=[]
attributions = (
    attributions.cpu().numpy()
)  # Move the attributions from the GPU to the CPU, and turn then into numpy arrays for future processing

# %% [markdown] tags=[]
# Here is an example for an image, and its corresponding attribution.


# %% tags=[]
from captum.attr import visualization as viz
import numpy as np


def visualize_attribution(attribution, original_image):
    attribution = np.transpose(attribution, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))

    viz.visualize_image_attr_multiple(
        attribution,
        original_image,
        methods=["original_image", "heat_map"],
        signs=["all", "absolute_value"],
        show_colorbar=True,
        titles=["Image", "Attribution"],
        use_pyplot=True,
    )


# %% tags=[]
for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_attribution(attr, im)

# %% [markdown]
#
# The attributions are shown as a heatmap. The closer to 1 the pixel value, the more important this attribution method thinks that it is.
# As you can see, it is pretty good at recognizing the number within the image.
# As we know, however, it is not the digit itself that is important for the classification, it is the color!
# Although the method is picking up really well on the region of interest, it would be difficult to conclude from this that it is the color that matters.


# %% [markdown]
# Something is slightly unfair about this visualization though.
# We are visualizing as if it were grayscale, but both our images and our attributions are in color!
# Can we learn more from the attributions if we visualize them in color?
# %%
def visualize_color_attribution(attribution, original_image):
    attribution = np.transpose(attribution, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_image)
    ax1.set_title("Image")
    ax1.axis("off")
    ax2.imshow(np.abs(attribution) / np.max(np.abs(attribution)))
    ax2.set_title("Attribution")
    ax2.axis("off")
    plt.show()


for attr, im, lbl in zip(attributions, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% [markdown]
# We get some better clues when looking at the attributions in color.
# The highlighting doesn't just happen in the region with number, but also seems to happen in a channel that matches the color of the image.
# Just based on this, however, we don't get much more information than we got from the images themselves.
#
# If we didn't know in advance, it is unclear whether the color or the number is the most important feature for the classifier.
# %% [markdown]
#
# ### Changing the baseline
#
# Many existing attribution algorithms are comparative: they show which pixels of the input are responsible for a network output *compared to a baseline*.
# The baseline is often set to an all 0 tensor, but the choice of the baseline affects the output.
# (For an interactive illustration of how the baseline affects the output, see [this Distill paper](https://distill.pub/2020/attribution-baselines/))
#
# You can change the baseline used by the `integrated_gradients` object.
#
# Use the command:
# ```
# ?integrated_gradients.attribute
# ```
# To get more details about how to include the baseline.
#
# Try using the code below to change the baseline and see how this affects the output.
#
# 1. Random noise as a baseline
# 2. A blurred/noisy version of the original image as a baseline.

# %% [markdown]
# <div class="alert alert-block alert-info"><h4>Task 2.3: Use random noise as a baseline</h4>
#
# Hint: `torch.rand_like`
# </div>

# %% tags=["task"]
# Baseline
random_baselines = ...  # TODO Change
# Generate the attributions
attributions_random = integrated_gradients.attribute(...)  # TODO Change
attributions_random = attributions_random.cpu().numpy()

# Plotting
for attr, im, lbl in zip(attributions_random, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% tags=["solution"]
#########################
# Solution for task 2.3 #
#########################
# Baseline
random_baselines = torch.rand_like(x)
# Generate the attributions
attributions_random = integrated_gradients.attribute(
    x, target=y, baselines=random_baselines
)
attributions_random = attributions_random.cpu().numpy()
# Plotting
for attr, im, lbl in zip(attributions_random, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h4>Task 2.4: Use a blurred image a baseline</h4>
#
# Hint: `torchvision.transforms.functional` has a useful function for this ;)
# </div>

# %% tags=["task"]
# TODO Import required function

# Baseline
blurred_baselines = ...  # TODO Create blurred version of the images
# Generate the attributions
attributions_blurred = integrated_gradients.attribute(...)  # TODO Fill

attributions_blurred = attributions_blurred.cpu().numpy()
# Plotting
for attr, im, lbl in zip(attributions_blurred, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% tags=["solution"]
#########################
# Solution for task 2.4 #
#########################
from torchvision.transforms.functional import gaussian_blur

# Baseline
blurred_baselines = gaussian_blur(x, kernel_size=(5, 5))
# Generate the attributions
attributions_blurred = integrated_gradients.attribute(
    x, target=y, baselines=blurred_baselines
)

attributions_blurred = attributions_blurred.cpu().numpy()

# Plotting
for attr, im, lbl in zip(attributions_blurred, x.cpu().numpy(), y.cpu().numpy()):
    print(f"Class {lbl}")
    visualize_color_attribution(attr, im)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h4> Questions </h4>
# <ul>
# <li>What baseline do you like best so far? Why?</li>
# <li>Why do you think some baselines work better than others?</li>
# <li>If you were to design an ideal baseline, what would you choose?</li>
# </ul>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>BONUS Task: Using different attributions.</h2>
#
#
# [`captum`](https://captum.ai/tutorials/Resnet_TorchVision_Interpret) has access to various different attribution algorithms.
#
# Replace `IntegratedGradients` with different attribution methods. Are they consistent with each other?
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 2</h2>
# Put up your sticky note when you've reached this point!
#
# At this point we have:
#
# - Loaded a classifier that classifies MNIST-like images by color, but we don't know how!
# - Tried applying Integrated Gradients to find out what the classifier is looking at - with little success.
# - Discovered the effect of changing the baseline on the output of integrated gradients.
#
# Coming up in the next section, we will learn how to create counterfactual images.
# These images will change *only what is necessary* in order to change the classification of the image.
# We'll see that using counterfactuals we will be able to disambiguate between color and number as an important feature.
# </div>

# %% [markdown]
# # Part 3: Train a GAN to Translate Images
#
# To gain insight into how the trained network classifies images, we will use [Discriminative Attribution from Counterfactuals](https://arxiv.org/abs/2109.13412), a feature attribution with counterfactual explanations methodology.
# This method employs a StarGAN to translate images from one class to another to make counterfactual explanations.
#
# **What is a counterfactual?**
#
# You've learned about adversarial examples in the lecture on failure modes. These are the imperceptible or noisy changes to an image that drastically changes a classifier's opinion.
# Counterfactual explanations are the useful cousins of adversarial examples. They are *perceptible* and *informative* changes to an image that change a classifier's opinion.
#
# In the image below you can see the difference between the two. In the first column are (non-color) MNIST images along with their classifications, and in the second column are counterfactual explanations to *change* that class. You can see that in both cases a human being would (hopefully) agree with the new classification. By comparing the two columns, we can therefore begin to define what makes each digit special.
#
# In contrast, the third and fourth columns show an MNIST image and a corresponding adversarial example. Here the network returns a prediction that most human beings (who aren't being facetious) would strongly disagree with.
#
# <img src="assets/ce_vs_ae.png" width=50% />
#
# **Counterfactual synapses**
#
# In this example, we will train a StarGAN network that is able to take any of our special MNIST images and change its class.
# %% [markdown] tags=[]
# ### The model
# ![stargan.png](assets/stargan.png)
#
# In the following, we create a [StarGAN model](https://arxiv.org/abs/1711.09020).
# It is a Generative Adversarial model that is trained to turn one class of images X into a different class of images Y.
#
# - The generator - this will be the bulk of the model, and will be responsible for transforming the images: we're going to use a `UNet`
# - The style encoder - this will be responsible for encoding the style of the image: we're going to use a `DenseModel`
# - The discriminator - this will be responsible for telling the difference between real and fake images: we're going to use a `DenseModel`
#
# Let's start by creating these!
# %%
from dlmbl_unet import UNet
from torch import nn


class Generator(nn.Module):

    def __init__(self, generator, style_encoder):
        super().__init__()
        self.generator = generator
        self.style_encoder = style_encoder

    def forward(self, x, y):
        """
        x: torch.Tensor
            The source image
        y: torch.Tensor
            The style image
        """
        style = self.style_encoder(y)
        # Concatenate the style vector with the input image
        style = style.unsqueeze(-1).unsqueeze(-1)
        style = style.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, style], dim=1)
        return self.generator(x)


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 3.1: Create the models</h3>
#
# We are going to create the models for the generator, discriminator, and style mapping.
#
# Given the Generator structure above, fill in the missing parts for the unet and the style mapping.
# %% tags=["task"]
style_size = 3
unet_depth = ...  # TODO Choose a depth for the UNet
style_encoder = DenseModel(
    input_shape=..., num_classes=...  # How big is the style space?
)
unet = UNet(depth=..., in_channels=..., out_channels=..., final_activation=nn.Sigmoid())

generator = Generator(unet, style_encoder=style_encoder)
# %% tags=["solution"]
# Here is an example of a working setup! Note that you can change the hyperparameters as you experiment.
# Choose your own setup to see what works for you.
style_size = 3
style_encoder = DenseModel(input_shape=(3, 28, 28), num_classes=3)
unet = UNet(depth=2, in_channels=6, out_channels=3, final_activation=nn.Sigmoid())
generator = Generator(unet, style_encoder=style_encoder)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Hyper-parameter choices</h3>
# <ul>
# <li>Are any of the hyperparameters above constrained in some way?</li>
# <li>What would happen if you chose a depth of 10 for the UNet?</li>
# <li>Is there a minimum size for the style space? Why or why not?</li>
# </ul>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 3.2: Create the discriminator</h3>
#
# We want the discriminator to be like a classifier, so it is able to look at an image and tell not only whether it is real, but also which class it came from.
# The discriminator will take as input either a real image or a fake image.
# Fill in the following code to create a discriminator that can classify the images into the correct number of classes.
# </div>
# %% tags=["task"]
discriminator = DenseModel(input_shape=..., num_classes=...)
# %% tags=["solution"]
discriminator = DenseModel(input_shape=(3, 28, 28), num_classes=4)
# %% [markdown]
# Let's move all models onto the GPU
# %%
generator = generator.to(device)
discriminator = discriminator.to(device)

# %% [markdown] tags=[]
# ## Training a GAN
#
# Training an adversarial network is a bit more complicated than training a classifier.
# For starters, we are simultaneously training two different networks that work against each other.
# As such, we need to be careful about how and when we update the weights of each network.
#
# We will have two different optimizers, one for the Generator and one for the Discriminator.
#
# %%
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
# %% [markdown] tags=[]
#
# There are also two different types of losses that we will need.
# **Adversarial loss**
# This loss describes how well the discriminator can tell the difference between real and generated images.
# In our case, this will be a sort of classification loss - we will use Cross Entropy.
# <div class="alert alert-block alert-warning">
# The adversarial loss will be applied differently to the generator and the discriminator! Be very careful!
# </div>
# %%
adversarial_loss_fn = nn.CrossEntropyLoss()

# %% [markdown] tags=[]
#
# **Cycle/reconstruction loss**
# The cycle loss is there to make sure that the generator doesn't output an image that looks nothing like the input!
# Indeed, by training the generator to be able to cycle back to the original image, we are making sure that it makes a minimum number of changes.
# The cycle loss is applied only to the generator.
#
# %%
cycle_loss_fn = nn.L1Loss()

# %% [markdown] tags=[]
# To load the data as batches, with shuffling and other useful features, we will use a `DataLoader`.
# %%
from torch.utils.data import DataLoader

dataloader = DataLoader(
    mnist, batch_size=32, drop_last=True, shuffle=True,
)  # We will use the same dataset as before


# %% [markdown] tags=[]
# As we stated earlier, it is important to make sure when each network is being trained when working with a GAN.
# Indeed, if we update the weights at the same time, we may lose the adversarial aspect of the training altogether, with information leaking into the generator or discriminator causing them to collaborate when they should be competing!
# `set_requires_grad` is a function that allows us to determine when the weights of a network are trainable (if it is `True`) or not (if it is `False`).
# %%
def set_requires_grad(module, value=True):
    """Sets `requires_grad` on a `module`'s parameters to `value`"""
    for param in module.parameters():
        param.requires_grad = value


# %% [markdown] tags=[]
# Another consequence of adversarial training is that it is very unstable.
# While this instability is what leads to finding the best possible solution (which in the case of GANs is on a saddle point), it can also make it difficult to train the model.
# To force some stability back into the training, we will use Exponential Moving Averages (EMA).
#
# In essence, each time we update the generator's weights, we will also update the EMA model's weights as an average of all the generator's previous weights as well as the current update.
# A certain weight is given to the previous weights, which is what ensures that the EMA update remains rather smooth over the training period.
# Each epoch, we will then copy the EMA model's weights back to the generator.
# This is a common technique used in GAN training to stabilize the training process.
# Pay attention to what this does to the loss during the training process!
# %%
from copy import deepcopy


def exponential_moving_average(model, ema_model, beta=0.999):
    """Update the EMA model's parameters with an exponential moving average"""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(beta).add_((1 - beta) * param.data)


def copy_parameters(source_model, target_model):
    """Copy the parameters of a model to another model"""
    for param, target_param in zip(
        source_model.parameters(), target_model.parameters()
    ):
        target_param.data.copy_(param.data)


# %%
generator_ema = Generator(deepcopy(unet), style_encoder=deepcopy(style_encoder))
generator_ema = generator_ema.to(device)


# %% [markdown] tags=[]
# <div class="alert alert-banner alert-info"><h4>Task 3.3: Training!</h4>
# You were given several different options in the training code below. In each case, one of the options will work, and the other will not.
# Comment out the option that you think will not work.
# <ul>
#   <li>Choose the values for <code>set_requires_grad</code>. Hint: which part of the code is training the generator? Which part is training the discriminator</li>
#   <li>Choose the values of <code>set_requires_grad</code>, again. Hint: you may want to switch</li>
#   <li>Choose the sign of the discriminator loss. Hint: what does the discriminator want to do?</li>
#   <li>Apply the EMA update. Hint: which model do you want to update? You can look again at the code we wrote above.</li>
# </ul>
# Let's train the StarGAN one batch a time.
# While you watch the model train, consider whether you think it will be successful at generating counterfactuals in the number of steps we give it. What is the minimum number of iterations you think are needed for this to work, and how much time do yo uthink it will take?
# </div>
# %% [markdown] tags=[]
# Once you're happy with your choices, run the training loop! &#x1F682; &#x1F68B; &#x1F68B; &#x1F68B;
# %% tags=["task"]


losses = {"cycle": [], "adv": [], "disc": []}

for epoch in range(15):
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
        x = x.to(device)
        y = y.to(device)
        # get the target y by shuffling the classes
        # get the style sources by random sampling
        random_index = torch.randperm(len(y))
        x_style = x[random_index].clone()
        y_target = y[random_index].clone()

        # TODO - Choose an option by commenting out what you don't want
        ############
        # Option 1 #
        ############
        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        ############
        # Option 2 #
        ############
        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)

        optimizer_g.zero_grad()
        # Get the fake image
        x_fake = generator(x, x_style)
        # Try to cycle back
        x_cycled = generator(x_fake, x)
        # Discriminate
        discriminator_x_fake = discriminator(x_fake)
        # Losses to  train the generator

        # 1. make sure the image can be reconstructed
        cycle_loss = cycle_loss_fn(x, x_cycled)
        # 2. make sure the discriminator is fooled
        adv_loss = adversarial_loss_fn(discriminator_x_fake, y_target)

        # Optimize the generator
        (cycle_loss + adv_loss).backward()
        optimizer_g.step()

        # TODO - Choose an option by commenting out what you don't want
        ############
        # Option 1 #
        ############
        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        ############
        # Option 2 #
        ############
        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)
        #
        optimizer_d.zero_grad()
        #
        discriminator_x = discriminator(x)
        discriminator_x_fake = discriminator(x_fake.detach())

        # TODO - Choose an option by commenting out what you don't want
        # Losses to train the discriminator
        # 1. make sure the discriminator can tell real is real
        # 2. make sure the discriminator can tell fake is fake
        ############
        # Option 1 #
        ############
        real_loss = adversarial_loss_fn(discriminator_x, y)
        fake_loss = -adversarial_loss_fn(discriminator_x_fake, y_target)
        ############
        # Option 2 #
        ############
        real_loss = adversarial_loss_fn(discriminator_x, y)
        fake_loss = adversarial_loss_fn(discriminator_x_fake, y_target)
        #
        disc_loss = (real_loss + fake_loss) * 0.5
        disc_loss.backward()
        # Optimize the discriminator
        optimizer_d.step()

        losses["cycle"].append(cycle_loss.item())
        losses["adv"].append(adv_loss.item())
        losses["disc"].append(disc_loss.item())

        # EMA update
        # TODO - perform the EMA update
        ############
        # Option 1 #
        ############
        exponential_moving_average(generator, generator_ema)
        ############
        # Option 2 #
        ############
        exponential_moving_average(generator_ema, generator)
    # Copy the EMA model's parameters to the generator
    copy_parameters(generator_ema, generator)
    # Save the model
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_ema": generator_ema.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch,
            "losses": losses,
        },
        f"extras/checkpoints/stargan_epoch_{epoch}.pth",
    )
# %% tags=["solution"]
losses = {"cycle": [], "adv": [], "disc": []}
for epoch in range(15):
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
        x = x.to(device)
        y = y.to(device)
        # get the target y by shuffling the classes
        # get the style sources by random sampling
        random_index = torch.randperm(len(y))
        x_style = x[random_index].clone()
        y_target = y[random_index].clone()

        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        optimizer_g.zero_grad()
        # Get the fake image
        x_fake = generator(x, x_style)
        # Try to cycle back
        x_cycled = generator(x_fake, x)
        # Discriminate
        discriminator_x_fake = discriminator(x_fake)
        # Losses to  train the generator

        # 1. make sure the image can be reconstructed
        cycle_loss = cycle_loss_fn(x, x_cycled)
        # 2. make sure the discriminator is fooled
        adv_loss = adversarial_loss_fn(discriminator_x_fake, y_target)

        # Optimize the generator
        (cycle_loss + adv_loss).backward()
        optimizer_g.step()

        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad()
        #
        discriminator_x = discriminator(x)
        discriminator_x_fake = discriminator(x_fake.detach())
        # Losses to train the discriminator
        # 1. make sure the discriminator can tell real is real
        real_loss = adversarial_loss_fn(discriminator_x, y)
        # 2. make sure the discriminator can tell fake is fake
        fake_loss = -adversarial_loss_fn(discriminator_x_fake, y_target)
        #
        disc_loss = (real_loss + fake_loss) * 0.5
        disc_loss.backward()
        # Optimize the discriminator
        optimizer_d.step()

        losses["cycle"].append(cycle_loss.item())
        losses["adv"].append(adv_loss.item())
        losses["disc"].append(disc_loss.item())
        exponential_moving_average(generator, generator_ema)
    # Copy the EMA model's parameters to the generator
    copy_parameters(generator_ema, generator)
    # Save the model
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_ema": generator_ema.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch,
            "losses": losses,
        },
        f"extras/checkpoints/stargan_epoch_{epoch}.pth",
    )


# %% [markdown] tags=[]
# Once training is complete, we can plot the losses to see how well the model is doing.
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot(losses["cycle"])
ax1.set_title("Cycle loss")
ax2.plot(losses["adv"])
ax2.set_title("Adversarial loss")
ax3.plot(losses["disc"])
ax3.set_title("Discriminator loss")
plt.show()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the losses look like what you expected? </li>
# <li> How do these losses differ from the losses you would expect from a classifier? </li>
# <li> Based only on the losses, do you think the model is doing well? </li>
# </ul>

# %% [markdown] tags=[]
# We can also look at some examples of the images that the generator is creating.
# %%
idx = 0
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(x[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[0].set_title("Input image")
axs[1].imshow(x_style[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[1].set_title("Style image")
axs[2].imshow(x_fake[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[2].set_title("Generated image")
axs[3].imshow(x_cycled[idx].cpu().permute(1, 2, 0).detach().numpy())
axs[3].set_title("Cycled image")

for ax in axs:
    ax.axis("off")
plt.show()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h2>Checkpoint 3</h2>
# You've now learned the basics of what makes up a StarGAN, and details on how to perform adversarial training.
# The same method can be used to create a StarGAN with different basic elements.
# For example, you can change the architecture of the generators, or of the discriminator to better fit your data in the future.
#
# You know the drill... put up your sticky note when you have arrived here!
# </div>

# %% [markdown] tags=[]
# # Part 4: Evaluating the GAN and creating Counterfactuals

# %% [markdown] tags=[]
# GANs are hard... and yours might not have worked out! If you've made it this far I hope you tried a few times 😉.

# %% [markdown] tags=[]
# ## Creating counterfactuals
#
# The first thing that we want to do is make sure that our GAN is able to create counterfactual images.
# To do this, we have to create them, and then pass them through the classifier to see if they are classified correctly.
#
# First, let's get the test dataset, so we can evaluate the GAN on unseen data.
# Then, let's get four prototypical images from the dataset as style sources.

# %% Loading the test dataset
test_mnist = ColoredMNIST("extras/data", download=True, train=False)
prototypes = {}


for i in range(4):
    options = np.where(test_mnist.conditions == i)[0]
    # Note that you can change the image index if you want to use a different prototype.
    image_index = 0
    x, y = test_mnist[options[image_index]]
    prototypes[i] = x

# %% [markdown] tags=[]
# Let's have a look at the prototypes.
# %%
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
for i, ax in enumerate(axs):
    ax.imshow(prototypes[i].permute(1, 2, 0))
    ax.axis("off")
    ax.set_title(f"Prototype {i}")

# %% [markdown]
# Now we need to use these prototypes to create counterfactual images!
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4.1: Create counterfactuals</h3>
# In the below, we will store the counterfactual images in the `counterfactuals` array.
#
# <ul>
# <li> Create a counterfactual image for each of the prototypes. </li>
# <li> Classify the counterfactual image using the classifier. </li>
# <li> Store the source and target labels; which is which?</li>
# </ul>
# %% tags=["task"]
num_images = 1000
random_test_mnist = torch.utils.data.Subset(
    test_mnist, np.random.choice(len(test_mnist), num_images, replace=False)
)
counterfactuals = np.zeros((4, num_images, 3, 28, 28))

predictions = []
source_labels = []
target_labels = []
with torch.inference_mode():
    for i, (x, y) in tqdm(enumerate(random_test_mnist), total=num_images):
        for lbl in range(4):
            # TODO Create the counterfactual
            x_fake = generator(x.unsqueeze(0).to(device), ...)
            # TODO Predict the class of the counterfactual image
            pred = model(...)

            # TODO Store the source and target labels
            source_labels.append(...)  # The original label of the image
            target_labels.append(...)  # The desired label of the counterfactual image
            # Store the counterfactual image and prediction
            counterfactuals[lbl][i] = x_fake.cpu().detach().numpy()
            predictions.append(pred.argmax().item())
# %% tags=["solution"]
num_images = 1000
random_test_mnist = torch.utils.data.Subset(
    test_mnist, np.random.choice(len(test_mnist), num_images, replace=False)
)
counterfactuals = np.zeros((4, num_images, 3, 28, 28))

predictions = []
source_labels = []
target_labels = []
with torch.inference_mode():
    for i, (x, y) in tqdm(enumerate(random_test_mnist), total=num_images):
        for lbl in range(4):
            # Create the counterfactual
            x_fake = generator(
                x.unsqueeze(0).to(device), prototypes[lbl].unsqueeze(0).to(device)
            )
            # Predict the class of the counterfactual image
            pred = model(x_fake)

            # Store the source and target labels
            source_labels.append(y)  # The original label of the image
            target_labels.append(lbl)  # The desired label of the counterfactual image
            # Store the counterfactual image and prediction
            counterfactuals[lbl][i] = x_fake.cpu().detach().numpy()
            predictions.append(pred.argmax().item())

# %% [markdown] tags=[]
# Let's plot the confusion matrix for the counterfactual images.
# %%
cf_cm = confusion_matrix(target_labels, predictions, normalize="true")
sns.heatmap(cf_cm, annot=True, fmt=".2f")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> How well is our GAN creating counterfactual images? </li>
# <li> Does your choice of prototypes matter? Why or why not? </li>
# </ul>
# </div>

# %% [markdown] tags=[]
# Let's also plot some examples of the counterfactual images.

# %%
for i in np.random.choice(range(num_images), 4):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for j, ax in enumerate(axs):
        ax.imshow(counterfactuals[j][i].transpose(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Class {j}")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li>Can you easily tell which of these images is the original, and which ones are the counterfactuals?</li>
# <li>What is your hypothesis for the features that define each class?</li>
# </ul>
# </div>

# %% [markdown]
# At this point we have:
# - A classifier that can differentiate between image of different classes
# - A GAN that has correctly figured out how to change the class of an image
#
# Let's try putting the two together to see if we can figure out what exactly makes a class.
#
# %%
target_class = 0
batch_size = 4
batch = [random_test_mnist[i] for i in range(batch_size)]
x = torch.stack([b[0] for b in batch])
y = torch.tensor([b[1] for b in batch])
x_fake = torch.tensor(counterfactuals[target_class, :batch_size])
x = x.to(device).float()
y = y.to(device)
x_fake = x_fake.to(device).float()

# Generated attributions on integrated gradients
attributions = integrated_gradients.attribute(x, baselines=x_fake, target=y)


# %% Another visualization function
def visualize_color_attribution_and_counterfactual(
    attribution, original_image, counterfactual_image
):
    attribution = np.transpose(attribution, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))
    counterfactual_image = np.transpose(counterfactual_image, (1, 2, 0))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    ax0.imshow(original_image)
    ax0.set_title("Image")
    ax0.axis("off")
    ax1.imshow(counterfactual_image)
    ax1.set_title("Counterfactual")
    ax1.axis("off")
    ax2.imshow(np.abs(attribution) / np.max(np.abs(attribution)))
    ax2.set_title("Attribution")
    ax2.axis("off")
    plt.show()


# %%
for idx in range(batch_size):
    print("Source class:", y[idx].item())
    print("Target class:", target_class)
    visualize_color_attribution_and_counterfactual(
        attributions[idx].cpu().numpy(), x[idx].cpu().numpy(), x_fake[idx].cpu().numpy()
    )
# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the attributions explain the differences between the images and their counterfactuals? </li>
# <li> What happens when the "counterfactual" and the original image are of the same class? Why do you think this is? </li>
# <li> Do you have a more refined hypothesis for what makes each class unique? </li>
# </ul>
# </div>
# %% [markdown]
# By now you will have hopefully noticed that it isn't the exact color of the image that determines its class, but that two images with a very similar color can be of different classes!
#
# Here are two examples of image-counterfactual-attribution triplets.
# You'll notice that they are *very* similar in every way! But one set is different classes, and one set is the same class!
#
# ![same_class](assets/same_class.png)
# ![diff_class](assets/diff_class.png)
#
# We are missing a crucial step of the explanation pipeline: a quantification of how the class changes over the interpolation.
#
# In the lecture, we used the attribution to act as a mask, to gradually go from the original image to the counterfactual image.
# This allowed us to classify all of the intermediate images, and learn how the class changed over the interpolation.
# Here we have a much simpler task so we have some advantages:
# - The counterfactuals are perfect! They already change the bare minimum (trust me).
# - The changes are not objects, but colors.
# As such, we will do a much simpler linear interpolation between the images.
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4.2: Interpolation</h3>
# Let's interpolate between the original image and the counterfactual image.
# We will create 10 images in between the two, and classify them.
# </div>
# %%
num_interpolations = 15
alpha = np.linspace(0, 1, num_interpolations + 2)[1:-1]
interpolated_images = [
    alpha[i] * x_fake + (1 - alpha[i]) * x for i in range(num_interpolations)
]
interpolated_images = torch.stack(interpolated_images)
interpolated_classifications = [
    model(interpolated_images[idx].to(device)) for idx in range(num_interpolations)
]
# %%
# Plot the results
idx = 0
fig, axs = plt.subplots(
    batch_size, num_interpolations + 2, figsize=(30, 2 * batch_size)
)
for idx in range(batch_size):
    # Plot the original image
    axs[idx, 0].imshow(np.transpose(x[idx].cpu().squeeze().numpy(), (1, 2, 0)))
    axs[idx, 0].axis("off")
    # Use the class as the title
    axs[idx, 0].set_title(f"Image: y={y[idx].item()}")
    # Plot the counterfactual image
    axs[idx, -1].imshow(np.transpose(x_fake[idx].cpu().squeeze().numpy(), (1, 2, 0)))
    axs[idx, -1].axis("off")
    # Use the target class as the title
    axs[idx, -1].set_title(f"CF: y={target_class}")
    for i, ax in enumerate(axs[idx][1:-1]):
        ax.imshow(
            np.transpose(interpolated_images[i][idx].cpu().squeeze().numpy(), (1, 2, 0))
        )
        ax.axis("off")
        classification = torch.softmax(interpolated_classifications[i][idx], dim=0)
        # Plot the classification as the title in order source classification | target classification
        ax.set_title(
            f"{classification[y[idx]].item():.2f} | {classification[target_class].item():.2f}"
        )
# %% [markdown]
# Take some time to look at the plot we just made.
# On the very left are the images we randomly chose - each image's class is shown in the title.
# On the very right are the counterfactual images, all of them made with the same prototype as a style source - the target class is shown in the title.
# In between are the interpolated images - their title shows their classification as "source classification | target classification".
# This is a lot to take in, so take your time! Once you're ready, we can move on to the questions.
# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the images change smoothly from one class to another? </li>
# <li> Can you see any patterns in the changes? </li>
# <li> What happens when the original image and the counterfactual image are of the same class? </li>
# <li> Based on this, would you trust this classifier on unseen images more or less than you did before? </li>
# </ul>
# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 4</h2>
# At this point you have:
# <ul>
# <li> Created a StarGAN that can change the class of an image </li>
# <li> Evaluated the StarGAN on unseen data </li>
# <li> Used the StarGAN to create counterfactual images </li>
# <li> Used the counterfactual images to highlight the differences between classes </li>
# <li> Interpolated between the images to see how the classifier behaves </li>
# </ul>
# %% [markdown]
# # Part 5: Exploring the Style Space, finding the answer
# So color is important... but not always? What's going on!?
# There is a final piece of information that we can use to solve the puzzle: the style space.
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.1: Explore the style space</h3>
# Let's take a look at the style space.
# We will use the style encoder to encode the style of the images and then use PCA to visualize it.
# </div>

# %%
from sklearn.decomposition import PCA


styles = []
labels = []
for img, label in random_test_mnist:
    styles.append(
        style_encoder(img.unsqueeze(0).to(device)).cpu().detach().numpy().squeeze()
    )
    labels.append(label)

# PCA
pca = PCA(n_components=2)
styles_pca = pca.fit_transform(styles)

# Plot the PCA
markers = ["o", "s", "P", "^"]
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.scatter(
        styles_pca[np.array(labels) == i, 0],
        styles_pca[np.array(labels) == i, 1],
        marker=markers[i],
        label=f"Class {i}",
    )
plt.legend()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.2: Adding color to the style space</h3>
# We know that color is important. Does interpreting the style space as colors help us understand better?
#
# Let's use the style space to color the PCA plot.
# (Note: there is no code to write here, just run the cell and answer the questions below)
# </div>
# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>🚨WARNING🚨</b> Pay attention to the markers' shapes in the legend and on the plot to avoid confusion when reading the plots (in particular, in the next plot. colour carries meaning different to that of the markers' shapes).
# </div>

# %%
styles = np.array(styles)
normalized_styles = (styles - np.min(styles, axis=1, keepdims=True)) / np.ptp(
    styles, axis=1, keepdims=True
)

# Plot the PCA again!
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.scatter(
        styles_pca[np.array(labels) == i, 0],
        styles_pca[np.array(labels) == i, 1],
        c=normalized_styles[np.array(labels) == i],
        marker=markers[i],
        label=f"Class {i}",
    )
plt.legend()
plt.show()
# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the colors match those that you have seen in the data?</li>
# <li> Can you see any patterns in the colors? Is the space smooth, for example?</li>
# </ul>
# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5.3: Using the images to color the style space</h3>
# Finally, let's just use the colors from the images themselves!
# The maximum value in the image (since they are "black-and-color") can be used as a color!
#
# Let's get that color, then plot the style space again.
# (Note: once again, no coding needed here, just run the cell and think about the results with the questions below)
# </div>
# %%
colors = np.array([np.max(x.numpy(), axis=(1, 2)) for x, _ in random_test_mnist])

# Plot the PCA again!
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.scatter(
        styles_pca[np.array(labels) == i, 0],
        styles_pca[np.array(labels) == i, 1],
        c=colors[np.array(labels) == i],
        marker=markers[i],
        label=f"Class {i}",
    )
plt.legend()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
# <li> Do the colors match those that you have seen in the data?</li>
# <li> Can you see any patterns in the colors?</li>
# <li> Can you guess what the classes correspond to?</li>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 5</h2>
# You have:
# <ul>
# <li> Created a StarGAN that can change the class of an image </li>
# <li> Evaluated the StarGAN on unseen data </li>
# <li> Used the StarGAN to create counterfactual images </li>
# <li> Used the counterfactual images to highlight the differences between classes </li>
# <li> Used the style space to understand the differences between classes </li>
# </ul>
#
# If you have any questions, feel free to ask them in the chat!
# And check the Solutions exercise for a definite answer to how these classes are defined!
#
# %% [markdown]
# We did a lot of work to try to interpret what was going on in this dataset.
# But, is this work always necessary?
#
# Sometimes, the data is in a format that is already amenable to interpretation, if we try a little bit harder than just looking at a few images.
# Let's try using the exact same code we used on the style space, but directly on the images themselves.

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Bonus Task: Exploring the image space</h3>
# Let's use PCA to visualize the images in the dataset, using the colors of the images themselves.
# </div>
# %%
images = []
for img, label in random_test_mnist:
    images.append(img.cpu().detach().numpy().squeeze())
images = np.array(images)

pca = PCA(n_components=2)
images_pca = pca.fit_transform(images.reshape(len(images), -1))

# Plot the PCA
markers = ["o", "s", "P", "^"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
for i in range(4):
    ax1.scatter(
        images_pca[np.array(labels) == i, 0],
        images_pca[np.array(labels) == i, 1],
        marker=markers[i],
        label=f"Class {i}",
    )
    ax1.set_title("PCA of images, colored by class")
    ax2.scatter(
        images_pca[np.array(labels) == i, 0],
        images_pca[np.array(labels) == i, 1],
        c=colors[np.array(labels) == i],
        marker=markers[i],
        label=f"Class {i}",
    )
    ax2.set_title("PCA of images, colored by image color")
plt.legend()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Questions</h3>
# <ul>
#   <li> Is this easier or harder to interpret than the style space? </li>
#  <li> Can you think of something else to plot that would be even more interpretable? </li>
# </ul>
# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 6</h2>
# Congratulations! You have made it to the end of the exercise!
#
# # Bonus!
# If you have extra time, you can try to break the StarGAN!
# There are a lot of little things that we did to make sure that it runs correctly - but what if we didn't?
# Some things you might want to try:
# <li> What happens if you change the learning rates? </li>
# <li> What happens if you add a Sigmoid activation to the output of the style encoder? </li>
# See what else you can think of, and see how finnicky training a GAN can be!

# %% [markdown] tags=["solution"]
# The colors for the classes are sampled from matplotlib colormaps! They are the four seasons: spring, summer, autumn, and winter.
# Check your style space again to see if you can see the patterns now!

# %% tags=["solution"]
# Let's plot the colormaps
import matplotlib as mpl
import numpy as np


def plot_color_gradients(cmap_list):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
        ax.text(
            -0.01,
            0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


plot_color_gradients(["spring", "summer", "autumn", "winter"])
