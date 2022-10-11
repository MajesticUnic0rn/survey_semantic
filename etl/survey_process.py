import pandas as pd
import numpy as np
from autocorrect import spell # autocorrect information
import spacy # might useless - but is a useful package to implement < - 
from tqdm.auto import tqdm
import helper
from helper import * #problematic for import all - but most of the functions I use within?
from classifier import TopicClassifier
import argparse

from utils import get_custom_logger
logger=get_custom_logger()

#args set tags
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="survey processor"
    )

    #parser.add_argument('-t','--tags', help="Number 1", type=[]) future feature if they want to add another set of tags
    parser.add_argument('-i','--input', help="input location", type=str)
    parser.add_argument('-o','--output', help="output location", type=str)
    args = parser.parse_args()
    # input 
    
    #input_survey=pd.read_csv('../data/WebsiteSatisfaction.csv')
    #start to rename the csv hard coding - throw exception if the required schema in the csv doesnt exist
    logger.info(f'reading {args.input}')
    input_survey=pd.read_csv(args.input)
    #schema check? 
    #first implementation of pipe for pandas
    logger.info(f'processing data')
    input_survey.dropna(inplace=True)
    input_processed = (input_survey.
    pipe(spell_chcker,['Comments']).
    pipe(text_blob_polarity_semantic_eng,['Comments_spell_checked']).
    pipe(time_stamp_cleaner, 'ActivityCompleted')
    )

    logger.info(f'running topic classifier')
    tags = ["User Experience", "customer service", "equipment availability", 
    "pricing error",
    "incorrect information", 
    "login",
    "transaction",
    "complexity"]

    zero_shot=TopicClassifier(tags) #model init first time is ruff.
    input_processed["topic"] = input_processed['Comments_spell_checked'].apply(zero_shot.top_prediction) 
    # is there a way to shrink the topic classifier time?
    #export util - to whatever filesource for now its CSV fixed
    #input_survey.to_csv('../data/cleaned_survey.csv')
    logger.info(f'outputing survey processing results')
    input_processed.to_csv(args.output)




