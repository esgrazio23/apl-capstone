import pandas as pd
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import time 

print('LOADED PACKAGES')
df = pd.read_csv("tweets_2020_cleaned.csv", lineterminator='\n', index_col=0) # set dtype on import 
df = df.dropna()
tweets = df["tweet"].astype(str).tolist()

print('LOADED DATA')
# define model params
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

    
# helper functions for running the pipeline
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
        
def run_batch(chunk):
    """given a list chunk, pass it to the sentiment pipeline and return labels and probabilities"""
    batch = tokenizer(chunk, padding=True, truncation=True, max_length=512, is_split_into_words=False, return_tensors="pt")

    with torch.no_grad():
        outputs=model(**batch)
        #print(outputs)
        probabilities=F.softmax(outputs.logits, dim=1)
        #print(probabilities)
        labels=torch.argmax(probabilities, dim=1)
        #print(labels) # 1 is positive, 0 is negative
    return labels, probabilities


print('BUILDING SENTIMENTS')
start_time = time.time()
labels = []    
confidence = []
# HF pipeline code to process entire list in chunks 
for chunk in chunks(tweets, 100):
    new_labels, new_confidence = run_batch(chunk)
    labels.extend(new_labels.tolist())
    confidence.extend([max(c).item() for c in new_confidence])
    print("%.2f%% complete" % (100 * len(labels) / len(tweets)))
    
end_time = time.time()
elapsed_time = end_time - start_time
print("time to classify all examples", elapsed_time)


# append results to input data and export to csv
label_series = pd.Series(labels)
conf_series = pd.Series(confidence)
df["sentiment"] = label_series
df["confidence"] = conf_series
df.to_csv("tweets_2020_sa.csv")
    
print("END OF SCRIPT")

