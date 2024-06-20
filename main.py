from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.svm import SVC


from information_system import RestaurantInfoBot
from reservation_system import ReservationSystem
from smalltalk_chat_bot import SmallTalkChatBot


class TextCategorizer:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        # Extract features and labels
        y , X = self.load_training_set("category_TR.csv")

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        svm_classifier = SVC(kernel='linear', probability=True)
        vectorizer=TfidfVectorizer(max_features=50)
        # Build a pipeline with CountVectorizer and Multinomial Naive Bayes classifier
        self.model = make_pipeline(vectorizer, svm_classifier)

        # Train the model
        self.trainModel(X,y)

    def trainModel(self,X,y):
        self.model.fit(X,y)

    def evaluate_model(self):
        self.model.fit(self.X_train, self.y_train)
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report

    def get_answer(self, question):
        # Predict the category of the question and get probabilities
        probabilities = self.model.predict_proba([question])[0]
        best_category_idx = np.argmax(probabilities)
        best_category_confidence = probabilities[best_category_idx]

        if best_category_confidence < self.threshold:
            return "error"
        return self.model.predict([question])[0]

    def get_categories(self):
        return ', '.join(set(self.y_test))

    def load_training_set(self,path = "small_talk_TR"):
        file_path = 'category_TR.csv'

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path,sep=";")

        # Extract the 'Category' and 'Sentence' columns into lists
        labels = df['Category'].tolist()
        texts = df['Sentence'].tolist()
        return labels, texts

print("Hello")

id_name_dict = {
    "123": "Alice",
    "456": "Bob",
    "789": "Charlie"
    # Add more ID-name pairs as needed
}

def collect_Name(id_name_dict):
        # Ask for user input
        user_input = input("Please enter your ID number. ")

        # Check the dictionary and respond
        if user_input in id_name_dict:
            print(f"Welcome back, {id_name_dict[user_input]}!")
            return id_name_dict[user_input]
        else:
            # Ask for the user's name and add it to the dictionary
            new_name = input("It seems you are new here. What's your name? ")
            id_name_dict[user_input] = new_name
            print(f"Nice to meet you, {new_name}! Your ID has been added to our system.")
            return new_name


# text_categorizer = TextCategorizer()
# accuracy, report = text_categorizer.evaluate_model()
# print("Accuracy:", accuracy)
# print("Classification Report:\n", report)
name = collect_Name(id_name_dict)
print("I am your friendly neighborhood chatbot.")
print("You can Chat with me make a reservation or ask me about the menu working hours and location of our restaurant. :-)")
print("If you wish to exit just say goodbye. :-)")
print(f"{name} how may i assist you today?")

text_categorizer = TextCategorizer()
restaurantInfoBot = RestaurantInfoBot()
reservation_system = ReservationSystem()
smallTalkBot =  SmallTalkChatBot()

context = "none"

while True:

    user_input = input()
    if user_input.lower() == 'exit':
        print(f"Goodbye {name} it was nice chating with you.")
        break

    if user_input.lower() =='info':
        print(f"{name} I see you need information. You can ask me about our menu, loaction ,payment and working hours.")
        while True:
            user_input = input()
            if user_input.lower() == 'exit':
                print(f"Exiting information mode.")
                break

            responce, answer_category = restaurantInfoBot.generate_response(user_input)
            print(responce)
            if answer_category != 'error':
                print(f"Anything else i can assist you with {name}")
        continue

    if user_input.lower() =='small_t':
        print("Entering small talk mode. You can Greet me Compiment me talk about the Weather or your Weekend Plans or general inquary. Type exit leave.")
        while True:
            user_input = input()
            if user_input.lower() == 'exit':
                print(f"Exiting small talk mode.")
                break
            sentiment, responce = smallTalkBot.get_response(user_input)
            print(sentiment + " " + responce)
        continue

    answer = text_categorizer.get_answer(user_input)
    if answer == 'error':
        if context == "none":
            print(f"I am not sure how to answer that. I can help with questions about: {text_categorizer.get_categories()}.")
        if context == "information": answer = "information"
        if context == "small_talk" : answer = "small_talk"
    if answer == 'information':
        print(f"{name} I see you need information. You can ask me about our menu, loaction ,payment and working hours.")
        responce, answer_category = restaurantInfoBot.generate_response(user_input)
        print(responce)
        context = "information"
        if answer_category !='error':
            print(f"Anything else i can assist you with {name}")
    if answer == 'reservation':
        reservation_system.make_reservation(name)
        print(f"Anything else Ι can assist you with {name}")
    if answer == 'small_talk':
        sentiment, responce = smallTalkBot.get_response(user_input)
        print(sentiment +" "+responce)
        context = "small_talk"


