import streamlit as st
import torch

st.title("Hello World")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

st.write("using", device, "device")
