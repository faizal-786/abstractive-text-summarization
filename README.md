# Resume-and-CV-Summarization-and-Parsing-with-Spacy-in-Python
Resume and CV Summarization and Parsing with Spacy in Python


Data preparation is the most difficult task. Unless you have large dataset.

Our Proposed Methodology :
Recruitment is a $200 Billion industry globally, with millions of people uploading resumes and applying for jobs everyday on thousands of employment  platforms. Businesses have their openings listed on these platforms and job  seekers come apply. Every business has a dedicated recruitment department that  manually goes through the applicant resumes and extracts relevant data to see if  they are a fit. 
CV Parsing or Resume summarization technology converts an unstructured form  of resume data into a structured format. It could be boon to HR. With the help of  machine learning and deep learning, an accurate and faster system can be made  which can save time for HR to scan each resume manually. 
Resume parsing software helps store, organize, and analyze resume  data automatically to find the best candidate. This software eliminates  manual data entry by extracting candidate’s information intelligently and saves it  in pre-designed fields.
Our approach is using Named Entity Recognition (NER). 
It is used when we want a specific set of strings from the extracted regions. 
For example, say in technical proficiencies section, there are sub-sections like  Platforms, Frameworks, Languages, Tools and any company wants only  programming languages he is good at. These types of problems can be solved  using NER, before going into depth, let’s see what it is about, 
Named Entity Recognition is an algorithm where it takes a string of text as an input (either a paragraph or sentence) and identifies relevant nouns (people,  places, and organizations) and other specific words. For example, in a given  resume if you want to extract, only Name and Phone Number using NER would  make our job much easier. It can be achieved by deep learning. This is because of  a technique called word embeddings, which is capable of understanding the  semantic and syntactic relationship between words. There’s some pre processing involved for most of the programs that involve data, even this Resume  Parsing includes one. In most of the cases, resumes are saved as PDFs or DOCX,  hence to make it easy, in the first steps, we’ll convert the word document into a  variable. 

Implementation :

So, we are going to create a model using SpaCy which will extract the main points
from a resume. We are going to train the model on almost 300 resumes. After the
model is ready, we will extract the text from a new resume and pass it to the
model to get the summary.
We can download the dataset from here:
https://github.com/faizal-786/abstractive-text-summarization

We have imported the necessary libraries. 
 import spacy 
import pickle 
import random
We will load the training data. The data consists of the contents of the resume  which is extracted from a PDF file, followed by a dictionary consisting of a label  and the start and end index of the value in the resume. In the example given  below Companies worked at is a custom label and there are multiple values for it  in the resume. 
train_data = pickle.load(open('train_data.pkl', 'rb')) train_data[0] 


We will first load a blank SpaCy English model. Then we will write a function that  will take the training data as the input. In the function, first, we will add a NER i.e.  Named Entity Recognition in the last position in the pipeline. Then we will add our  custom labels in the pipeline. 
Now we are going to prepare our data for training. We disable all the pipeline components except NER. We are only going to train NER. We are going to train for  10 iterations. At each iteration, the training data is shuffled to ensure the model doesn’t make any generalizations based on the order of examples. 
We are again going to read the training data. Another technique to improve the  learning results is to set a dropout rate, a rate at which to randomly “drop”  individual features and representations. This makes it harder for the model to  memorize the training data. We have added a dropout of 0.2 which means that  each feature or internal representation has a 1/5 likelihood of being dropped. 
Lastly, we will train the model on our data. 

nlp = spacy.blank('en') 
def train_model(train_data): 
 if 'ner' not in nlp.pipe_names: 
 ner = nlp.create_pipe('ner') 
 nlp.add_pipe(ner, last = True)  
 for _, annotation in train_data: 
 for ent in annotation['entities']: 
 ner.add_label(ent[2])
nlp = spacy.blank('en') 
 other_pipes = [pipe for pipe in nlp.pipe_names if  pipe != 'ner'] 
 with nlp.disable_pipes(*other_pipes): # only  train NER 
 optimizer = nlp.begin_training() 
 for itn in range(10): 
 print("Statring iteration " + str(itn))  random.shuffle(train_data) 
 losses = {} 
 index = 0 
 for text, annotations in train_data:  try: 
 nlp.update( 
 [text], # batch of texts 
 [annotations], # batch of  annotations 
 drop=0.2, # dropout - make it  harder to memorise data 
 sgd=optimizer, # callable to  update weights 
 losses=losses) 
 except Exception as e: 
 pass   
 print(losses)
train_model(train_data) 
Statring iteration 0 
{'ner': 15037.69734973145} Statring iteration 1 
{'ner': 14219.407121717988} Statring iteration 2 
{'ner': 10966.300542596338} Statring iteration 3 
{'ner': 9323.783655686213} Statring iteration 4 
{'ner': 8716.684299001408} 
Statring iteration 5 
{'ner': 7539.511302326008} Statring iteration 6 
{'ner': 5957.294807362359} Statring iteration 7 
{'ner': 4725.2797832249025} Statring iteration 8 
{'ner': 5975.5969295029045} Statring iteration 9 
{'ner': 4244.961263079282}
The model will take a lot of time to train. So we are saving the model for further  use. 
nlp.to_disk('nlp_model') 
Now we will load the saved model into nlp_model. 
nlp_model = spacy.load('nlp_model') 
We can access first resume from our training data. Due to  
random.shuffle(train_data) used in the function train_model() we may get a  different resume at the first position. 
train_data[0][0] 


