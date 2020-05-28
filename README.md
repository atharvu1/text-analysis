
# text-analysis
## Setup

### Activating virualenv
In the project directory:

For macOS and linux <br/>
`source 'env/Srcipts/activate' `<br/>
For Windows<br/>
`.\env\Scripts\activate`<br/>

### Starting server
For macOS and linux<br/>
`python app.py`<br/>
For Windows<br/>
`py index.py`<br/>
This will initiate server on [localhost:5000](http://127.0.0.1:5000)<br/>
####  Program may give some error on starting server make sure you have all the packages and files installed in your workspace.

#### You can also download the nltk models directly from the app.py or from Python interpreter, add the below lines of code.
```
import nltk
nltk.download('model-name')
