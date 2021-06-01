import streamlit as st
from mega import Mega
import zipfile
import os

# Download model files if they don't exist #
if not os.path.isdir("./models"):
    mega = Mega()
    try:
        mega.download_url(
            "https://mega.nz/#!S4AGzQJD!UH7B5SV7DJSTqKvtbFKqFkjdAh60kpdhTk9WerI-Q1I"
        )
    except PermissionError:
        pass
    with zipfile.ZipFile("./maps_BtoA.zip", "r") as zip_ref:
        zip_ref.extractall("./models/maps_BtoA")

from streamlit_auxlib import *

# Web app formatting #
def display_random_example():
    images = generate_example()
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.header("Input")
        st.image(images["input"])
    with col2:
        st.header("Output")
        st.image(images["output"])
    with col3:
        st.header("Real Image")
        st.image(images["target"])


def generate_new_example(input_file):
    file_type = image_file.name.split(".")[-1]
    images = generate_example(input_image_file=input_file, file_type=file_type)

    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.header("Input")
        st.image(images["input"])
    with col2:
        st.header("Output")
        st.image(images["output"])


st.title("Pix2pix Maps to Satellite Demo")

example_type = st.selectbox(
    "Display a random example, or generate a new one by inputting a google maps image:",
    ("Random Example", "Generate Example"),
)

if example_type == "Random Example":
    display_random_example()

if example_type == "Generate Example":
    image_file = st.file_uploader("Upload a PNG or JPG image", type=([".jpg"]))
    st.write(
        "To generate a reasonable satellite image, the uploaded image must be "
        "be in google maps style and with no labels/text, 600x600 or larger with "
        "3 channels. The image should be zoomed in enough that individual roads are"
        " clearly visible. See the input images in the random examples as a "
        "reference. Generated image quality may vary dramatically if the input "
        "differs from these guidelines. "
    )
    st.write(
        "Here is one resource that can be used to generate the desired style of "
        "images: https://snazzymaps.com/style/24088/map-without-labels"
    )
    if image_file:
        generate_new_example(image_file)

st.header("Citation")
st.write(
    "This code entirely based on the pix2pix2016 paper, "
    "found here: https://arxiv.org/abs/1611.07004"
)
st.write(
    "@article{pix2pix2016,  \n"
    "title={Image-to-Image Translation with Conditional Adversarial Networks},  \n"
    "author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},  \n"
    "journal={arxiv},  \n"
    "year={2016}}  \n"
)
