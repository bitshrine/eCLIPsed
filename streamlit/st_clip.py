import streamlit as st
import pandas as pd
import numpy as np
import models
import torch
import PIL
from PIL import Image
import clip
import matplotlib.pyplot as plt

from pkg_resources import packaging

from setup import *
from console import *

###############
##!!WARNING!!##
###############
# install those before running:
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
# choose option:
# conda install --yes -c pytorch pytorch=1.7.1 torchvision [cudatoolkit=11.0 cpuonly]

st.title('Clip app')

# load model and requirements
with st_stdout("info"):
    fetch_LELSD_requirements()

from utils.stylegan2_utils import StyleGAN2SampleGenerator
import random

random.seed(2023)
seed = random.getrandbits(16)

@st.cache
def load_clip(model_name, device):
    return clip.load(clip_model_name, device=device)

@st.cache_data
def load_model(model_name):
    return fetch_model(model_name)

# choose number of samples
n_samples = 50000 

model_path = load_model('metfaces')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

G2 = models.get_model("stylegan2", model_path)

generator = StyleGAN2SampleGenerator(G=G2,device=device, only_w=True)
samples = generator.generate_batch(
    seed=seed,
    batch_size=n_samples,
    return_image=False,
    return_all_layers=False,
    return_style=True,
)

# get N random images
N = 10
maxSize = len(samples['ws']-3)
images = []
for i in range(N):
    index = round(random.random() * maxSize)
    img = generator.generate_image_from_ws(samples['ws'][index:index+2])
    img = img[0]
    images.append(img)

# choose clip's model
clip_model_name = st.sidebar.selectbox(
    "Choose the CLIP model you want.",
    clip.available_models(),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_clip(clip_model_name, device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

st.write("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
st.write("Input resolution:", input_resolution)
st.write("Context length:", context_length)
st.write("Vocab size:", vocab_size)

# labels to feed to CLIP
labels = ["a man", "a woman", "a child"]


# wtf
# compute the probs of each image with each label
text = clip.tokenize(labels).to(device)
probabilities = []

for img in images:
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probabilities.append(probs[0].tolist())

# show the images and their probabilities
count = len(labels)
fig, ax = plt.subplots(figsize=(20, 14))
ax.imshow(probabilities, vmin=0, vmax=1)
# ax.colorbar()
ax.set_yticks(range(count))
ax.set_yticklabels(labels, fontsize=18)
ax.set_xticks([])
i = 0
for image in images:
    i += 1
    ax.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(count):
    for y in range(N):
        ax.text(y, x, f"{probabilities[y][x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
    ax.spines[side].set_visible(False)

ax.set_xlim([-0.5, count - 0.5])
ax.set_ylim([count + 0.5, -2])

ax.set_title("Probability of each label on the image", size=20)

st.pyplot(fig)
