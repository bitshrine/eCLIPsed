import streamlit as st
import pandas as pd
import numpy as np
import models
import torch
import PIL


from setup import *
from console import *

st.title('Generate an impage')

with st_stdout("info"):
    fetch_LELSD_requirements()

from utils.stylegan2_utils import StyleGAN2SampleGenerator
import random

random.seed(2023)
seed = random.getrandbits(16)

n_samples = 50000 

model_path = fetch_model('metfaces')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


G2 = models.get_model("stylegan2" ,model_path)

generator = StyleGAN2SampleGenerator(G=G2,device=device, only_w=True)
samples = generator.generate_batch(
    seed=seed,
    batch_size=n_samples,
    return_image=False,
    return_all_layers=False,
    return_style=True,

)

st.write('The size of the torch is', samples['ws'].size())
maxSize = len(samples['ws']-3)
random = round(random.random() * maxSize)
label = "give me a random number to change the picture, between 1 and " + str(maxSize)
index = st.number_input(label, 0, maxSize, step=1, value=random)
img = generator.generate_image_from_ws(samples['ws'][index:index+2])
img = img[0]
st.image(img)

