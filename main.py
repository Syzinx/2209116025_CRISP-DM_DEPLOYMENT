import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import pickle
from streamlit_option_menu import *
from function import *



df = pd.read_csv('Data Cleaned.csv')
st.title('causes that influence the amount of tips at restaurants')

with st.sidebar :
    selected = option_menu('A WAITER TIPS',['Introducing','Data Distribution','Relation','Composition & Comparison','Predict'],default_index=0)

if (selected == 'Introducing'):
    st.write("""
    The biggest influence on the tip amount given is the total on the bill
    """)
    st.title("Correlation between Bill Amount and Tip")
    subplot()
    st.write("From the scatterplot correlation, we can see that there is a positive correlation between total_bill and tip, which means that the bigger the total bill, the bigger the tip given. However, keep in mind that correlation does not mean causation. Although there is a positive correlation between total_bill and tip, we cannot conclude that a large total bill causes a large tip.")
    st.title("Other influences also have a big impact")

    st.write("Select Other:")

    impact_types = ["Based On Gender", "Based On Day", "Based On Meal times"]
    impact_type = st.selectbox("impact Type", impact_types)
    
    translate = st.checkbox("Translate to Indonesia")

    other_impact(impact_type, translate)

if (selected == 'Data Distribution'):
    st.header("Data Distribution")
    scatterplot_all_impacts(df)


if (selected == 'Relation'):
    st.title('Relations')
    heatmap(df)

if (selected == 'Composition & Comparison'):
    st.title('Composition')
    compositionAndComparison(df)

if (selected == 'Predict'):
    st.title('Let\'s Predict!')
    predict()

