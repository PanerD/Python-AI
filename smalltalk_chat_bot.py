import random

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer, sentiment_analyzer


class SmallTalkChatBot:

    def __init__(self, threashold = 0.5):
        self.threashold = threashold
        # Create sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Extract features and labels
        y, X = self.load_training_set("small_talk_TR.csv")

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        svm_classifier = SVC(kernel='linear', probability=True)
        vectorizer = TfidfVectorizer(max_features = 40)
        vectorizer.fit(self.X_train)
        num_features = len(vectorizer.vocabulary_)
        # Build a pipeline with CountVectorizer and Multinomial Naive Bayes classifier
        self.model = make_pipeline(vectorizer, svm_classifier)

        # Train the model
        self.trainModel(X,y)


    def trainModel(self, X, y):
        self.model.fit(X, y)

    def evaluate_model(self):
        self.model.fit(self.X_train, self.y_train)
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report

    def get_response(self, input_text):
        # Predict the category of the question and get probabilities
        probabilities = self.model.predict_proba([input_text])[0]
        best_category_idx = np.argmax(probabilities)
        best_category_confidence = probabilities[best_category_idx]

        predicted_category = self.model.predict([input_text])[0]
        if best_category_confidence < self.threashold:
            predicted_category = "default"

        res = self.generate_response(predicted_category)

        # Integrade sentiment_analyzer
        sentiment_score = self.sentiment_analyzer.polarity_scores(input_text)
        sent = self.determine_sentiment(sentiment_score)

        return res, sent

    def generate_response(self, category):
        # Generate answers based on category
        responses = {
            "Greeting": [ "Hello! " ,"Hi there! How can I assist you today?" ,"Greetings! It's nice to chat with you." ,"Hey!","Good to see you!"],
            "Weekend Plans": ["Weekends are a great time to relax and do things you enjoy."
                ,"Planning something exciting for the weekend? It's always good to have something to look forward to."
                ,"Weekend plans can be such a highlight of the week."
                ,"I hope your weekend is shaping up to be a great one!"],
            "Weather": ["The weather is looking great today!"
                ,"It's quite sunny today."
                ,"Expect some rain in the evening."],
            "Compliment": ["Thank you! I'm here to help in any way I can."
                ,"That's so kind of you to say! I appreciate it."
                ,"I'm flattered, thank you! How can I assist you further?"
                ,"Your compliment made my circuits feel warm! How else can I be of service?"
                ,"I'm glad to be of help. Your positive feedback is encouraging!"],
            "General Inquiry": ["Everything is running smoothly on my end! How about you?"
                ,"I'm here and ready to help! What's on your mind?"
                ,"As a chatbot, I'm always good. What about you? Anything exciting happening?"
                ,"I'm functioning as expected. Let's make this conversation interesting!"
                ,"I'm here to assist you. Feel free to share or ask anything you'd like."],
            "default": ["I'm not sure how to respond to that. I am a baby unlike my big brother chatGPT."]
        }
        return random.choice(responses.get(category, responses["default"]))

    def determine_sentiment(self, sentiment_score):
        # Determine sentiment
        if sentiment_score['compound'] >= 0.05:
            return "That is great :-) "
        elif sentiment_score['compound'] <= -0.05:
            return "I am sorry to hear that :-( "
        else:
            return ""

    def load_training_set(self, path="small_talk_TR"):
        file_path = 'small_talk_TR.csv'

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, sep=";")

        # Extract the 'Category' and 'Sentence' columns into lists
        labels = df['Category'].tolist()
        texts = df['Sentence'].tolist()
        return labels, texts


# chatbot = SmallTalkChatBot()
# accuracy, report = chatbot.evaluate_model()
# print("Accuracy:", accuracy)
# print("Classification Report:\n", report)

# while True:
#     input_text = input()
#     if input_text == 'exit':
#         break
#     response, sentiment = chatbot.get_response(input_text)
#     print(sentiment + response)
