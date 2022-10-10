import streamlit as st

# EDA Pkgs

import pandas as pd
import numpy as np

# Data Viz Pkg

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


def main():
    activities = ['Visualize']
    choice = st.sidebar.selectbox('Select Activities', activities)
    
    if choice == 'Visualize':
        st.subheader('Visualize Survey Results')

        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
        #TODO data schema checker to throw error
        if data is not None:
            input_survey = pd.read_csv(data)
            st.dataframe(input_survey.head())

            mean_nps_time=input_survey.groupby('ActivityCompleted').mean()
            fig = go.Figure(
                [go.Scatter(x=mean_nps_time.index, y=mean_nps_time['nps_score'])],
                layout=go.Layout(
                title=go.layout.Title(text="Average Satisfaction Score")
                )  
            )
            fig.show()

            sentiment_chart=input_survey.groupby('Sentiment').size()
            fig = px.pie(values=sentiment_chart,names=sentiment_chart.index,title='Semantic Count Overall')
            fig.show()


            nps_best=' '.join(input_survey[input_survey['nps_score'] > 3 ].nlargest(n=50, columns=['Comments_spell_checked_txtblob_polarity','Comments_spell_checked_txtblob_polarity_subjectivity'])['Comments'])
            nps_worse=' '.join(input_survey[input_survey['nps_score'] < 3 ].nsmallest(n=50, columns=['Comments_spell_checked_txtblob_polarity','Comments_spell_checked_txtblob_polarity_subjectivity'])['Comments'])


            st.write('## Here are your worst comments so far') # in regards to topic/keywords
            st.write('')
            st.write('## Here are your best comments so far')# in regards to topic/keywords
            st.write('')


            #generate word cloud based on sub/polarity and filter on topic option


            if st.checkbox('Show Shape'):
                st.write(input_survey.shape)

            if st.checkbox('Show Columns'):
                all_columns = input_survey.columns.to_list()
                st.write(all_columns)

            if st.checkbox('Summary'):
                st.write(input_survey.describe())

            if st.checkbox('Show Selected Columns'):
                selected_columns = st.multiselect('Select Columns',
                        all_columns)
                new_df = input_survey[selected_columns]
                st.dataframe(new_df)

            if st.checkbox('Correlation Plot(Seaborn)'):
                st.write(sns.heatmap(input_survey.corr(), annot=True))
                st.pyplot()

            if st.checkbox('Pie Plot'):
                all_columns = input_survey.columns.to_list()
                column_to_plot = st.selectbox('Select 1 Column',
                        all_columns)
                pie_plot = \
                    input_survey[column_to_plot].value_counts().plot.pie(autopct='%1.1f%%'
                        )
                st.write(pie_plot)
                st.pyplot()
    