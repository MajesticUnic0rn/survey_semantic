import pandas as pd
import numpy as np
from autocorrect import spell # autocorrect information
import spacy # might useless - but is a useful package to implement < - 
from tqdm.auto import tqdm
import helper
from helper import *
from classifier import TopicClassifier



#make it args based script - input csv - then output csv location
input_survey=pd.read_csv('../data/WebsiteSatisfaction.csv')
#start to rename the csv hard coding - throw exception if the required schema in the csv doesnt exist
#drop the na and tell the business to get fucked 
input_survey.dropna(inplace=True)

#first implementation of pipe for pandas
input_processed = (input_survey.
pipe(spell_chcker,['Comments']).
pipe(text_blob_polarity_semantic_eng,['Comments_spell_checked']).
pipe(time_stamp_cleaner, 'ActivityCompleted')
)

tags = ["User Experience", "customer service", "equipment availability", 
"pricing error",
"incorrect information", 
"login",
"transaction",
"complexity"]
zero_shot=TopicClassifier(tags) #model init first time is ruff.
input_processed["topic"] = input_processed['Comments_spell_checked'].apply(zero_shot.top_prediction)
input_survey.to_csv('../data/cleaned_survey.csv')





