import pandas as pd
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import time

print('LOADED PACKAGES')
df = pd.read_csv("./zsl_subsets/tweets_2018_harassment_cleaned.csv", lineterminator='\n', index_col=0) # set dtype on import
df=df.dropna()
tweets = df["tweet"].astype(str).tolist()

print('LOADED DATA')
# define model params
classifier = pipeline('zero-shot-classification')
input_text = tweets
candidate_labels = ['military', 'air force']

    
print('BEGIN MODEL TRAINING')
start_time = time.time()
result = classifier(input_text, candidate_labels, multi_label=True) # does not require chunking like sa_pipeline
end_time = time.time()
elapsed_time = end_time - start_time

print("time to classify all examples", elapsed_time)

# append results to input data and export to csv
result_df = pd.DataFrame(result)
result_df = pd.concat([df, result_df], axis=1)
result_df.to_csv('tweets_2018_harassment_zsl.csv', index=False)


print("END OF SCRIPT")

