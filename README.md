## Welcome to Disaster Response Application

#### A machine learning pipeline to classify messages to be sent to appropriate disaster relief agency

The application home page gives an overview of training dataset, including the distributions of message categories, as well as the distribution of message genres

To get the message categories, enter the message in the input field and click "Classify Message" button, the result will show as highlighted in the result list.

### Prerequisites

- Python
- Scikit Learn - machine learning library
- Plotly - front-end graph library ML models

### ETL Pipeline

"ETL Pipeline Preparation.ipynb" python notebook file contains the data analysis to prepare for build ETL pipeline.

The workspace/data/process_data.py cleans the messages and categories dataset then stores in a Sqlite database
To run the ETL pipeline, run the following command from data directory:

```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```


### ML pipeline

"ML Pipeline Preparation.ipynb" python notebook file contains the data analysis and code to prepare for build ML pipeline.

The train_classifier.py builds a text processing and machine learning pipeline
To run the ML pipeline to train ML model, from model directory run:

```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
This will also output the model classification report which include precision, recall, and F-score for each classification categories.

### Flask Web App
The app directory contains the front-end user interface to display graph and to allow user to input message and get categories that the message belongs to

### Run the app:
From the app directory run: 
```
python run.py
```
