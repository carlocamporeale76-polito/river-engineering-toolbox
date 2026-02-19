import streamlit as st

st.set_page_config(page_title="River Engineering Toolbox", layout="wide")

st.title("River Engineering Toolbox")
st.write(
    """
This web companion provides engineering calculators associated with the textbook.

Use the left sidebar to select a chapter/tool.
"""
)

st.markdown(
    """
### Available tools
- **Chapter 2 – Sediment Mechanics**
  - Surface-based bedload transport for mixtures (Ashida–Michiue, Parker, Wilcock–Crowe)
"""
)
