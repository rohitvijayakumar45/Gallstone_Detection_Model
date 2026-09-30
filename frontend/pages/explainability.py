import sys
import os
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    from frontend.app import page_explainability
    page_explainability()
except ImportError:
    import streamlit as st
    st.error("Could not find 'frontend.app'. Please ensure you are running from the project root.")
