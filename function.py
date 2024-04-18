import streamlit as st
import seaborn as sns
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from sklearn.naive_bayes import GaussianNB
import os


df = pd.read_csv('Data Cleaned.csv')

def translate_text(text, target_language='id'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def translate_list_of_texts(text_list, target_language='id'):
    translator = Translator()
    translated_texts = [translator.translate(text, dest=target_language).text for text in text_list]
    return translated_texts

def subplot():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x='total_bill', y='tip', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'}, ax=ax)
    plt.title('Correlation between Bill Amount and Tip')
    plt.xlabel('Total Bill ($)')
    plt.ylabel('Tip ($)')
    plt.grid(True)
    st.pyplot(fig)

def other_impact(impact_type, translate):
    if impact_type == "Based On Gender":
        st.subheader("Gender Impact")
        gender_info = "The average total tip given by male customers tends to be higher than that of female customers. In the sample data analyzed, it is indicated that men tend to give larger tips than women"
        if translate:
            gender_info = translate_text(gender_info)
        st.markdown(gender_info)

    elif impact_type == "Based On Day":
        st.subheader("Day Impact")
        gender_info = "in general Sundays have a higher tip amount compared to other days. However, on Saturday, there were many anomalous values ​​that were higher than average. This shows a trend where some customers tend to give larger tips on Saturdays. The effect of the day on the tip amount may be related to different activities or eating out habits on certain days, such as weekends or holidays."
        if translate:
            gender_info = translate_text(gender_info)
        st.markdown(gender_info)

    elif impact_type == "Based On Meal times":
        st.subheader("Meal times Impact")
        gender_info = "Based on available data, it appears that the average tip amount at dinner time tends to be higher than at lunch time. This may be due to several factors, including cultural customs that emphasize the importance of dinner as a time for socializing or also because dinner often involves fancier dishes or more menu choices, which may encourage customers to leave larger tips in appreciation of services provided."
        if translate:
            gender_info = translate_text(gender_info)
        st.markdown(gender_info)

def scatterplot_all_impacts(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='total_bill', y='tip', hue='sex', style='size', markers=['o', 's', 'D', '^', 'X'], data=df, ax=ax)
    ax.set_title('Scatter Plot of Total Bill vs Tip with Different Impacts')
    ax.set_xlabel('Total Bill ($)')
    ax.set_ylabel('Tip ($)')
    ax.grid(True)
    ax.legend(title='Gender', loc='upper left')
    st.pyplot(fig)
    translate = st.checkbox("Translate to Indonesia")
    text = 'You can see the scatter plot which shows that the average total tip given by male customers tends to be higher than that of female customers. and you can also see that size also has quite an influence where when many people at one table tend to give bigger tips, at least in the data sample analyzed.'
    if translate:
        translated_text = translate_text(text)
        if translated_text:
            text = translated_text
    st.markdown(text)


def heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Heatmap of Correlation Matrix')
    st.pyplot(fig)
    translate = st.checkbox("Translate to Indonesia")
    text = 'You can conclude from the visual representation of the correlation matrix above that each cell (box) in the heatmap shows the level of correlation between a pair of variables; where the cell color is proportional to the correlation value. The color scale is on the right side, showing that red indicates a strong positive correlation (towards +1.0), and blue indicates a strong negative correlation (towards -1.0). Colors that are more neutral or closer to white show a weaker correlation.'
    if translate:
        translated_text = translate_text(text)
        if translated_text:
            text = translated_text
    st.markdown(text)


def compositionAndComparison (df):
# Hitung rata-rata fitur untuk setiap kelas
    df['sex'].replace({0: 'total_bill', 1: 'sex'}, inplace=True)
    class_composition = df.groupby('sex').mean()
    # Plot komposisi kelas
    plt.figure(figsize=(10, 6))
    sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu')
    plt.title('Composition for each impact')
    plt.xlabel('impact')
    plt.ylabel('Feature')
    st.pyplot(plt)
    translate = st.checkbox("Translate to Indonesia")
    text = 'As you can see the bar plot above shows the composition of a class which is taken from the average of each existing feature (column) and there is also a comparison of each feature used.'
    if translate:
        translated_text = translate_text(text)
        if translated_text:
            text = translated_text
    st.markdown(text)

def predict():
    total_bill = st.number_input('Total Bill', 0.00, 1000.00, step=0.01)
    sex_mapping = {'Male': 1, 'Female': 0}
    sex = st.radio('Sex', ('Male', 'Female'), format_func=lambda x: sex_mapping[x])
    size = st.number_input('Size', 1, 10)
    button = st.button('Predict')
    
    if button:
        # Membuat DataFrame user_data
        user_data = pd.DataFrame({
            'total_bill': [total_bill],
            'sex': [sex_mapping[sex]],
            'size': [size]
        })

        # Memuat model dari file dtc.joblib
        loaded_model = load('dtc.joblib')

        # Melakukan prediksi dengan model yang dimuat
        predicted_tip = loaded_model.predict(user_data)

        st.write(f"Predicted Tip Range: ${predicted_tip.min()} - ${predicted_tip.max()}")
