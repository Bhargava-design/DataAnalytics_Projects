#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import necessary Libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, render_template_string


# In[3]:


#Download NLTK data (if not already installed)
nltk.download('vader_lexicon')


# In[4]:


#Initialize the VADER Sentiment analyzer
sia = SentimentIntensityAnalyzer()

#Create a Flask Web Application
app = Flask(__name__)


# In[5]:


#Define the route for the feedback form
@app.route('/', methods=['GET', 'POST'])
def feedback_form():
    if request.method == "POST":
        #Get user feedback from the form
        feedback = request.form.get('user_feedback')
        
        #Save the feedback to a file (feedback.txt)
        with open('feedback.txt', 'a') as file:
            file.write(feedback + '\n')
            
        #Analyze the sentiment of the user's feedback
        sentiment, sentiment_score = analyze_sentiment(feedback)
        
        #Render the result page with sentiment information
        result_html = f"""
        <p> Your Feedback : {feedback} </p>
        <p> Sentiment: {sentiment} </p>
        <p> Sentiment Score : {sentiment_score} </p>
        <a href = "/"> Back to the feedback form
        """
        return result_html
    
    feedback_form_html = f"""
    <form method = "post" action="/" >
        <label for="user_feedback"> Enter your feedback here: </label>
        <input type =" text" name = "user_feedback" id= "user_feedback" required>
        <input type="submit" value ="submit">
    </form>
    """
    return feedback_form_html


# In[6]:


#Define a function for sentiment analysis
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    
    #determine sentiment weather it is negative or positive
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05 :
        sentiment = "Positive"
    elif compound_score <= -0.05 :
        sentiment = "Negative"
    else :
        sentiment = "Neutral"
        
    return sentiment, sentiment_score


# In[ ]:


#Run the flask app
app.run(debug=True, use_reloader=False)


# In[ ]:




