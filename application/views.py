from django.shortcuts import render
import pickle

model = pickle.load(open('pickle_files/kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('pickle_files/scaler.pkl', 'rb'))
pca = pickle.load(open('pickle_files/pca_model.pkl', 'rb'))
encoder1 = pickle.load(open('pickle_files/encoder1.pkl', 'rb'))
encoder2 = pickle.load(open('pickle_files/encoder2.pkl', 'rb'))
encoder3 = pickle.load(open('pickle_files/encoder3.pkl', 'rb'))
encoder4 = pickle.load(open('pickle_files/encoder4.pkl', 'rb'))

# Create your views here.
def home_page(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        flight_distance = int(request.POST.get('flight_distance'))
        wifi_service = int(request.POST.get('wifi_service'))
        departure_convenience = int(request.POST.get('departure_convenience'))
        online_booking = int(request.POST.get('online_booking'))
        food_drink = int(request.POST.get('food_drink'))
        online_boarding = int(request.POST.get('online_boarding'))
        seat_comfort = int(request.POST.get('seat_comfort'))
        entertainment = int(request.POST.get('entertainment'))
        onboard_service = int(request.POST.get('onboard_service'))
        leg_room = int(request.POST.get('leg_room'))
        baggage_handling = int(request.POST.get('baggage_handling'))
        checkin_service = int(request.POST.get('checkin_service'))
        inflight_service = int(request.POST.get('inflight_service'))
        cleanliness = int(request.POST.get('cleanliness'))
        departure_delay = int(request.POST.get('departure_delay'))
        arrival_delay = float(request.POST.get('arrival_delay'))
        Gender_Female, Gender_Male = encoder1.transform([[request.POST.get('gender')]]).toarray()[0]
        Customer_Type_Loyal_Customer, Customer_Type_disloyal_Customer = encoder2.transform([[request.POST.get('customer_type')]]).toarray()[0]
        Type_of_Travel_Business_travel, Type_of_Travel_Personal_Travel = encoder3.transform([[request.POST.get('type_of_travel')]]).toarray()[0]
        Class_Business, Class_Eco, Class_Eco_Plus = encoder4.transform([[request.POST.get('class')]]).toarray()[0]
        new_scaled_data = scaler.transform([[age, flight_distance, wifi_service, 
                           departure_convenience, online_booking,food_drink, online_boarding, 
                           seat_comfort, entertainment, onboard_service, leg_room, baggage_handling, checkin_service, inflight_service, 
                           cleanliness, departure_delay, arrival_delay, Gender_Female, Gender_Male, Customer_Type_Loyal_Customer, Customer_Type_disloyal_Customer, 
                           Type_of_Travel_Business_travel, Type_of_Travel_Personal_Travel, Class_Business, Class_Eco, Class_Eco_Plus]])
        new_pca_data = pca.transform(new_scaled_data)
        res = model.predict(new_pca_data)
        if res[0] == 0:
            res = 'Unsatisfied'
        else:
            res = 'Satisfied'
        return render(request, 'home_page.html', {'result':res})
    else:
        return render(request, 'home_page.html')

def about_page(request):
    return render(request, 'about_project.html')