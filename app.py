import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

st.header('This is a model that will descirbe what is in the image that you put in')

imageUpload = st.file_uploader('put your image here')

if imageUpload != None:

    image = Image.open(imageUpload)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.write(generated_caption)
    st.image(imageUpload)
