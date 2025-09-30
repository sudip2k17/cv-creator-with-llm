import streamlit as st
from llama_index import Document as LIDoc

st.title("LlamaIndex import test")
doc = LIDoc(text="Hello from LlamaIndex")
st.write("Created Document:", doc)
