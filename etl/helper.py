import pandas as pd
import numpy as np 
from autocorrect import Speller
from textblob import TextBlob
from typing import List

# copy that shit but you gotta set the deep vs shallow

def copy_df(df:pd.DataFrame):
   return df.copy(deep=False)

def remove_digits(input_df:pd.DataFrame,cols: List[str]=[])-> pd.DataFrame:
    data_copy = input_df
    for each_cols in cols:
        data_copy[each_cols+'_digits_removed']=data_copy[each_cols].str.replace('\d+','')
    return data_copy

# spell check the information given
# time breaking function
def spell_chcker(input_df:pd.DataFrame,cols: List[str]=[])-> pd.DataFrame:
    spell=Speller(fast=True)
    data_copy = input_df
    for each_cols in cols:
        data_copy[each_cols+"_spell_checked"] = [' '.join([spell(i) for i in x.split()]) for x in data_copy[each_cols]]
    return data_copy

#nlp feature eng 
def polarity_semantic_eng(input_df:pd.DataFrame,cols: List[str]=[])-> pd.DataFrame:
    data_copy = input_df
    sid = SIA()
    for each_cols in cols:
        data_copy[each_cols+'_polarity_score']=data_copy[each_cols].apply(polarity_check)
    return data_copy

# different polarity checker with additional feature of objective score
def text_blob_polarity_semantic_eng(input_df:pd.DataFrame,cols: List[str]=[])-> pd.DataFrame:
    data_copy = input_df
    for each_cols in cols:
        data_copy[each_cols+'_textblob_polarity_score']=data_copy[each_cols].apply(sentiment_check) # insert string into text blob inside
        data_copy[[each_cols+'_txtblob_polarity', each_cols+'_txtblob_polarity_subjectivity']] = pd.DataFrame(data_copy[each_cols+'_textblob_polarity_score'].tolist(), index=data_copy.index)
    return data_copy

#date time cleanup
def time_stamp_cleaner(input_df:pd.DataFrame ,column:str)-> pd.DataFrame:
    data_copy = input_df
    data_copy[column] = pd.to_datetime(data_copy[column], errors='coerce')
    data_copy[column] = data_copy[column].dt.date
    data_copy = data_copy.set_index(column)
    return data_copy
#extra TODO stuff:
#someone please teach me how to get this shit back into one of my functions instead of making 4 functions
def polarity_check(input_str):
    sid = SIA()
    return sid.polarity_scores(input_str)

def sentiment_check(input_str):
    return TextBlob(input_str).sentiment

#throw exception if .pipe has missing column to exit out schema checker on required fields that needs to be inside >
def column_check():
    return True

