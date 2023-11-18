from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch #pytorch
import torch.nn.functional as F

#Uses the sentimental analysis model to analyzer if the prompt is positive or negative
#Creates the object for the model, since theres no specified model it will invoke the default model
classifier= pipeline("sentiment-analysis")

#stores the response after invoking the model
res=classifier("I am having an amazing day")

print(res)

generator=pipeline("text-generation", model="distilgpt2")
#Stores the response of the invoked text generator function and sets the max length of its response
#to 30 characters and max number of lines generated to 2.
resp = generator(
    "In this course we will teach you how to",
    max_length=30,
  num_return_sequences=2,
)

print(resp)

#Manually selecting/setting up the model and its tokenizer
model_name="distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer= AutoTokenizer.from_pretrained(model_name)

#Performing sentimentak analysis with the selected model and tokenizer
classifier=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#Sequence that will be the training data
X_train= ["Ive been waiting for a huggingface course my whole life",
          "Python is great!"]

sent=classifier(X_train)
print("This is the sentimental analysis: ", sent)

####Below is an example of the logic that goes into what the pipeline is doing
#It will be useful to do this when wanting to fine tune our model

#Tokenizing the training data so that it can be used in the model
batch=tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    #passing the training batch through the model and recieving the outputs
    outputs = model(**batch)
    print(outputs)
    #Then using that output data to perform prdictions using the softmax activation function
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)


#Saves a model to be used for later
saved_directory = "saved"
tokenizer.save_pretrained(saved_directory)
model.save_pretrained(saved_directory)

tok = AutoTokenizer.from_pretrained(saved_directory)
mod = AutoModelForSequenceClassification.from_pretained(saved_directory)
