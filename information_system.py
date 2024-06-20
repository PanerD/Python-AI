import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class RestaurantInfoBot:
    def __init__(self, threshold=0.5):
        self.y,self.X = self.load_training_set()
        self.threshold = threshold  # Confidence threshold for matching categories

        # Creating and training the model
        self.model = self.create_model()

    def create_model(self):
        # Vectorize the questions and create a classifier pipeline
        vectorizer = TfidfVectorizer()
        svm_classifier = SVC(kernel='linear', probability=True)

        pipeline = make_pipeline(vectorizer, svm_classifier)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Train the model
        pipeline.fit(self.X_train, self.y_train)
        #pipeline.fit(self.X,self.y)

        return pipeline

    def get_answer(self, question):
        # Predict the category of the question and get probabilities
        probabilities = self.model.predict_proba([question])[0]
        best_category_idx = np.argmax(probabilities)
        best_category_confidence = probabilities[best_category_idx]

        if best_category_confidence < self.threshold:
            available_categories = ', '.join(set(self.y))
            #return f"I am not sure how to answer that. I can help with questions about: {available_categories}."
            return "error"

        return ["hours","location","menu","payment"][best_category_idx]

    def generate_response(self, question,name=''):
        # generate answers based on category
        responses = {
            "menu": [name+" We have all kinds of souvlakkia an pitogyra",name+" We have anything your heart desires.",
                     name+" We have everything and all the options so the answer to your question is yes"],
            "location": [name+" We are located in nowhere land.",
                         "Our address is in nowhere land ask my older brother google."],
            "hours": ["Our restaurant is open from 9 AM to 10 PM on weekdays, and from 10 AM to 11 PM on weekends.",
                      "Our restaurant is open 24/7."],
            "payment": [name+ " We will accept any payment", name+"We accept gold coins so what you ask is completely doable.",
                        "You are lucky you get your meal for free."],
            "error": [name+" No sure what you are asking you can asking me. If you need information you can ask me about menu , location , payment or hours "]
        }
        answerCategory = self.get_answer(question);
        return random.choice(responses.get(answerCategory)) , answerCategory

    def evaluate_model(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report

# Example usage
    def load_training_set(self,path = "small_talk_TR"):
        file_path = 'faq_TR.csv'

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path,sep=",")

        # Extract the 'Category' and 'Sentence' columns into lists
        labels = df['Category'].tolist()
        texts = df['Sentence'].tolist()
        return labels, texts


# bot = RestaurantInfoBot(threshold=0.6)
# # Evaluate the model
# accuracy, report = bot.evaluate_model()
# print("Accuracy:", accuracy)
# print("Classification Report:\n", report)
#
# question = ("What is on the menu ?")
# answer = bot.get_answer(question)
# print(bot.generate_response(question,"Bob"))
