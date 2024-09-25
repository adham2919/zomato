import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
import joblib
import streamlit as st




pipeline = joblib.load('model.pkl')

# Set up the Streamlit UI
st.title("Restaurant Prediction App")

cost_for_two = st.slider("Cost for Two", min_value=40, max_value=6000, value=1000)

book_online = st.selectbox("Book Online", ["yes", "no"])
online_order = st.selectbox("Online Order", ["yes", "no"])
location = st.selectbox("Location", ['Banashankari', 'Basavanagudi', 'Jayanagar', 'Kumaraswamy Layout',
                                      'Rajarajeshwari Nagar', 'Vijay Nagar', 'Mysore Road', 'Uttarahalli',
                                      'South Bangalore', 'Bannerghatta Road', 'Jp Nagar', 'Btm',
                                      'Kanakapura Road', 'Bommanahalli', 'Electronic City', 'Wilson Garden',
                                      'Shanti Nagar', 'Koramangala 5Th Block', 'Richmond Road', 'City Market',
                                      'Bellandur', 'Sarjapur Road', 'Marathahalli', 'Whitefield', 'Hsr',
                                      'Old Airport Road', 'Indiranagar', 'Koramangala 1St Block', 'East Bangalore',
                                      'Frazer Town', 'Mg Road', 'Brigade Road', 'Lavelle Road', 'Church Street',
                                      'Ulsoor', 'Residency Road', 'Shivajinagar', 'Infantry Road', 'St. Marks Road',
                                      'Cunningham Road', 'Race Course Road', 'Domlur', 'Koramangala 8Th Block', 'Ejipura',
                                      'Vasanth Nagar', 'Jeevan Bhima Nagar', 'Old Madras Road', 'Commercial Street',
                                      'Seshadripuram', 'Koramangala 6Th Block', 'Majestic', 'Langford Town',
                                      'Koramangala 7Th Block', 'Brookefield', 'Itpl Main Road, Whitefield',
                                      'Varthur Main Road, Whitefield', 'Koramangala 3Rd Block', 'Koramangala 2Nd Block',
                                      'Koramangala 4Th Block', 'Koramangala', 'Hosur Road', 'Banaswadi', 'North Bangalore',
                                      'Rt Nagar', 'Kammanahalli', 'Nagawara', 'Hennur', 'Hbr Layout', 'Kalyan Nagar',
                                      'Thippasandra', 'Cv Raman Nagar', 'Kaggadasapura', 'Kengeri', 'Rammurthy Nagar',
                                      'Sankey Road', 'Central Bangalore', 'Malleshwaram', 'Sadashiv Nagar',
                                      'Basaveshwara Nagar', 'Rajajinagar', 'New Bel Road', 'West Bangalore', 'Yeshwantpur',
                                      'Sanjay Nagar', 'Sahakara Nagar', 'Jalahalli', 'Hebbal',
                                      'Yelahanka', 'Magadi Road', 'Kr Puram'])

rest_type = st.selectbox("Restaurant Type", ['Casual Dining', 'Cafe, Casual Dining', 'Quick Bites',
                                               'Casual Dining, Cafe', 'Cafe', 'Quick Bites, Cafe',
                                               'Cafe, Quick Bites', 'Delivery', 'Mess', 'Dessert Parlor',
                                               'Bakery, Dessert Parlor', 'Pub', 'Bakery', 'Fine Dining',
                                               'Beverage Shop', 'Bar', 'Takeaway, Delivery', 'Sweet Shop',
                                               'Kiosk', 'Food Truck', 'Quick Bites, Dessert Parlor',
                                               'Beverage Shop, Quick Bites', 'Beverage Shop, Dessert Parlor',
                                               'Takeaway', 'Pub, Casual Dining', 'Casual Dining, Bar',
                                               'Dessert Parlor, Beverage Shop', 'Microbrewery, Casual Dining',
                                               'Sweet Shop, Quick Bites', 'Lounge', 'Bar, Casual Dining',
                                               'Food Court', 'Cafe, Bakery', 'Dhaba', 'Microbrewery', 'Pub, Bar',
                                               'Bakery, Quick Bites', 'Casual Dining, Pub', 'Lounge, Bar',
                                               'Dessert Parlor, Quick Bites', 'Casual Dining, Sweet Shop',
                                               'Casual Dining, Microbrewery', 'Lounge, Casual Dining',
                                               'Cafe, Food Court', 'Beverage Shop, Cafe', 'Cafe, Dessert Parlor',
                                               'Dessert Parlor, Cafe', 'Quick Bites, Sweet Shop',
                                               'Microbrewery, Pub', 'Quick Bites, Beverage Shop',
                                               'Food Court, Quick Bites', 'Dessert Parlor, Bakery', 'Club',
                                               'Quick Bites, Food Court', 'Bakery, Cafe', 'Pub, Cafe',
                                               'Casual Dining, Irani Cafee', 'Fine Dining, Lounge',
                                               'Quick Bites, Bakery', 'Bar, Quick Bites', 'Pub, Microbrewery',
                                               'Microbrewery, Lounge', 'Fine Dining, Microbrewery',
                                               'Fine Dining, Bar', 'Dessert Parlor, Kiosk', 'Cafe, Bar',
                                               'Casual Dining, Lounge', 'Dessert Parlor, Sweet Shop',
                                               'Food Court, Dessert Parlor', 'Microbrewery, Bar', 'Cafe, Lounge',
                                               'Confectionery', 'Bar, Pub', 'Lounge, Cafe', 'Club, Casual Dining',
                                               'Quick Bites, Mess', 'Quick Bites, Meat Shop',
                                               'Lounge, Microbrewery', 'Bakery, Food Court', 'Bar, Lounge',
                                               'Food Court, Beverage Shop', 'Food Court, Casual Dining'])

cuisines_count = st.selectbox("Cuisines Count", [1, 2, 3, 4, 5, 6, 7, 8])

food_type = st.selectbox("Food Type", ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
                                         'Drinks & Nightlife', 'Pubs and Bars'])
city = st.selectbox("City", ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
                               'Brigade Road', 'Brookefield', 'Btm', 'Church Street',
                               'Electronic City', 'Frazer Town', 'Hsr', 'Indiranagar',
                               'Jayanagar', 'Jp Nagar', 'Kalyan Nagar', 'Kammanahalli',
                               'Koramangala 4Th Block', 'Koramangala 5Th Block',
                               'Koramangala 6Th Block', 'Koramangala 7Th Block', 'Lavelle Road',
                               'Malleshwaram', 'Marathahalli', 'Mg Road', 'New Bel Road',
                               'Old Airport Road', 'Rajajinagar', 'Residency Road',
                               'Sarjapur Road', 'Whitefield'])

main_cuisine = st.selectbox("main_cuisines", ['multi-cuisine', 'South Indian', 'North Indian', 'Cafe', 'Bakery',
       'Pizza', 'Biryani', 'Street Food', 'Burger', 'Chinese',
       'Ice Cream', 'Fast Food', 'Beverages', 'Mithai', 'Italian',
       'Arabian', 'Vietnamese', 'Desserts', 'Andhra', 'Salad', 'Juices',
       'Mexican', 'Seafood', 'Rolls', 'Tibetan', 'Finger Food', 'Bengali',
       'Oriya', 'Kerala', 'Bohri', 'African', 'Rajasthani',
       'Healthy Food', 'Continental', 'Mangalorean', 'American',
       'Hyderabadi', 'Roast Chicken', 'Maharashtrian', 'European', 'Tea',
       'Mughlai', 'Asian', 'Konkan', 'Sandwich', 'Kebab', 'Modern Indian',
       'Bihari', 'Lebanese', 'Australian', 'Thai', 'Bbq', 'Portuguese',
       'Parsi', 'Japanese', 'North Eastern', 'Chettinad', 'Spanish',
       'Korean', 'Momos', 'Kashmiri', 'Gujarati', 'Lucknowi', 'Turkish',
       'Assamese', 'Tamil', 'Burmese', 'Coffee', 'French', 'Goan',
       'Bar Food', 'Mediterranean'])
rest_type_count = st.selectbox("rest_type_count", [1, 2])


input_data = {
    'cost_for_two': [cost_for_two], 
    'book_table': [book_online],      
    'location': [location],           
    'online_order': [online_order],    
    'city': [city],                    
    'food_type': [food_type],          
    'cuisines_count': [cuisines_count],  
    'rest_type': [rest_type],
    'rest_type_count':[rest_type_count],
    'main_cuisine':[main_cuisine]
    
}

input_df = pd.DataFrame(input_data)

if st.button("Predict"):
    # Make a prediction using the entire pipeline
    prediction = pipeline.predict(input_df)

    # Display the prediction
    st.success(f"The prediction is: {prediction[0]}")

# Add a reset button to clear inputs
if st.button("Reset"):
    st.experimental_rerun()
