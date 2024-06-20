import random
from datetime import datetime

class ReservationSystem:
    def __init__(self):
        self.reservations = []

    def is_full(self, date):
        # Replace this with actual logic to check reservations for the date
        return random.choice([True, False])

    def make_reservation(self,logedName):
        print("Welcome to our reservation system! (type 'exit' at any time to quit)")

        while True:
            # Getting the number of people
            people = input("Enter the number of people for the reservation: ")
            if people.lower() == 'exit':
                print("Reservation process terminated.")
                return

            while True:
                # Getting the reservation date
                date_str = input("Enter the reservation date (YYYY-MM-DD): ")
                if date_str.lower() == 'exit':
                    print("Reservation process terminated.")
                    return


                try:
                    if datetime.now() > datetime.strptime(date_str, "%Y-%m-%d"):
                        print("I guess you are a time traveler we can see if we have available table in the past. :-)")
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
                    continue

            # Check if the restaurant is full on that date
                if self.is_full(date):
                    print(f"Sorry, the restaurant is at full capacity on {date_str}. Please choose another day.")
                else:
                    break

            # Optionally getting the name
            name = input("Enter your name (optional, press enter to skip): ")
            if name.lower() == 'exit':
                print("Reservation process terminated.")
                return

            # Confirmation
            confirm = input(f"Confirm reservation for {people} people on {date_str} "
                            f"{'under the name ' + name if name else logedName} (yes/no): ").strip().lower()
            if confirm == 'exit':
                print("Reservation process terminated.")
                return
            elif confirm != 'yes':
                print("Reservation cancelled.")
                return
            elif confirm == 'yes':
                print(f"Reservation made.")

            # Creating the reservation
            reservation = {
                "people": people,
                "date": date,
                "name": name if name else "Anonymous"
            }

            self.reservations.append(reservation)
            print("Reservation successfully created!")
            break

    def show_reservations(self):
        print("Current Reservations:")
        for res in self.reservations:
            print(f"Date: {res['date'].strftime('%Y-%m-%d')}, People: {res['people']}, Name: {res['name']}")

# Usage
#reservation_system = ReservationSystem()
#reservation_system.make_reservation("Guest")
#reservation_system.show_reservations()
