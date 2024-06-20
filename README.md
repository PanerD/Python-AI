
# Chatbot 

This repository contains a chatbot system designed for restaurant-related interactions, including general information, reservation management, and small talk. The system is built using Python and leverages machine learning models for natural language processing and classification.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files and Directories](#files-and-directories)
- [Models](#models)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chatbot-repo.git
    ```
2. Change into the repository directory:
    ```bash
    cd chatbot-repo
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Information System**: Provides answers to frequently asked questions about the restaurant.
    ```python
    from information_system import RestaurantInfoBot

    info_bot = RestaurantInfoBot()
    question = "What are your opening hours?"
    response, category = info_bot.generate_response(question)
    print(response)
    ```

2. **Reservation System**: Manages restaurant reservations.
    ```python
    from reservation_system import ReservationSystem

    reservation_system = ReservationSystem()
    reservation_system.make_reservation("Guest")
    reservation_system.show_reservations()
    ```

3. **Small Talk Bot**: Engages in casual conversation with users.
    ```python
    from smalltalk_chat_bot import SmallTalkChatBot

    chat_bot = SmallTalkChatBot()
    input_text = "Hello, how are you?"
    response, sentiment = chat_bot.get_response(input_text)
    print(sentiment + response)
    ```

## Files and Directories

- `main.py`: Main entry point for the chatbot system, integrating all components.
- `information_system.py`: Contains the `RestaurantInfoBot` class for handling restaurant information queries.
- `reservation_system.py`: Contains the `ReservationSystem` class for managing reservations.
- `smalltalk_chat_bot.py`: Contains the `SmallTalkChatBot` class for small talk interactions.
- `requirements.txt`: Lists all dependencies required to run the chatbot.

## Models

The chatbot system uses Support Vector Machines (SVM) for text classification. The models are trained on specific datasets for different types of interactions:

- **Restaurant Information**: Uses a dataset of frequently asked questions and their categories.
- **Small Talk**: Uses a dataset of casual conversation examples.
- **Reservation System**: Logic-based and does not require a machine learning model.


## License

This project is licensed under the MIT License. 
