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
import datetime

def main(): 
    activities = ['Visualize']
    choice = st.sidebar.selectbox('Select Activities', activities)
    
    if choice == 'Visualize':
        st.subheader('Visualize Survey Results')
        #data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
        data = 'data/cleaned_data.csv'
        input_survey = pd.read_csv(data)
        filtered_df = df_filter('Move Sliders To Filter Time',input_survey)
        input_survey = filtered_df
    
        #TODO data schema checker to throw error
        if input_survey is not None: #if statement not needed if hard coded data file location
    
            col1, col2= st.columns(2)

            with col1:
                sentiment_chart=input_survey.groupby('Sentiment').size()
                fig = px.pie(values=sentiment_chart,names=sentiment_chart.index,title='Semantic Count Overall')
                st.plotly_chart(fig, use_container_width=True)

                st.write('Topic Modeling in Total')
                dfg = input_survey.groupby(["topic"]).count()
                fig = px.bar(dfg, x=dfg.index, y="ActivityCompleted")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                sentiment_chart=input_survey.groupby('topic').size()
                fig = px.pie(values=sentiment_chart,names=sentiment_chart.index,title='Topic Modeling Count Overall')
                st.plotly_chart(fig, use_container_width=True)
                st.write('Topic Modeling Crosstab with Sentiment')
                topic_graph=pd.crosstab(input_survey.topic, input_survey.Sentiment ,normalize = 'index')
                topic_graph.reset_index(inplace=True)
                fig = px.bar(topic_graph, x="topic", y=["Neutral", "Mixed","Positive" ,"Negative"])
                st.plotly_chart(fig, use_container_width=True)
            
            mean_nps_time=input_survey.groupby('ActivityCompleted').mean()
            fig = go.Figure(
            [go.Scatter(x=mean_nps_time.index, y=mean_nps_time['nps_score'])],
            layout=go.Layout(
            title=go.layout.Title(text="Average Satisfaction Score")
                    )  
                )
            st.plotly_chart(fig, use_container_width=True)
            
            #generate word cloud based on sub/polarity and filter on topic option
            if st.checkbox('Show Worst Subjective Comments'):
                st.write('## Here are your worst comments so far') # in regards to topic/keywords
                comment_worse_length = st.slider('Worst Comment Length', 10, 30, 40)
                nps_worse=' '.join(input_survey[input_survey['nps_score'] < 3 ].nsmallest(n=comment_worse_length, columns=['Comments_spell_checked_txtblob_polarity','Comments_spell_checked_txtblob_polarity_subjectivity'])['Comments'])
                st.write(f'{nps_worse}')
            if st.checkbox('Show Best Subjective Comments'):
                st.write('## Here are your best comments so far')# in regards to topic/keywords
                comment_best_length = st.slider('Best Comment Length', 10, 30, 40)
                nps_best=' '.join(input_survey[input_survey['nps_score'] > 3 ].nlargest(n=comment_best_length, columns=['Comments_spell_checked_txtblob_polarity','Comments_spell_checked_txtblob_polarity_subjectivity'])['Comments'])
                st.write(f'{nps_best}')

            if st.checkbox('Show Shape'):
                st.write(input_survey.shape)
            if st.checkbox('Show A Preview Of Data'):
                st.dataframe(input_survey.head())
            if st.checkbox('Show Columns'):
                all_columns = input_survey.columns.to_list()
                st.write(all_columns)
            if st.checkbox('Summary'):
                st.write(input_survey.describe())

def df_filter(message,df): #pray to god that your first in the index is a date time index column

        slider_1, slider_2 = st.slider('%s' % (message),0,len(df)-1,[0,len(df)-1],1)

        while len(str(df.iloc[slider_1][1]).replace('.0','')) < 4:
            df.iloc[slider_1,1] = '0' + str(df.iloc[slider_1][1]).replace('.0','')
            
        while len(str(df.iloc[slider_2][1]).replace('.0','')) < 4:
            df.iloc[slider_2,1] = '0' + str(df.iloc[slider_1][1]).replace('.0','')

        start_date = (str(df.iloc[slider_1][0]).replace('.0',''))
        end_date = (str(df.iloc[slider_2][0]))
        st.info('Start: **%s** End: **%s**' % (start_date,end_date))
        
        filtered_df = df.iloc[slider_1:slider_2+1][:].reset_index(drop=True)

        return filtered_df


if __name__ == '__main__':
    main()