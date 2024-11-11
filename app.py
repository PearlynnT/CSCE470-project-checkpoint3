from flask import Flask, request, render_template, jsonify
import re
import pandas as pd
import requests
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from dotenv import load_dotenv
from CountriesDictionary import Countries as cd
from geopy.geocoders import Nominatim

app = Flask(__name__)
load_dotenv()

GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY')
EXCHANGE_RATE_API_KEY = os.environ.get('EXCHANGE_RATE_API_KEY')
client_id = os.environ.get('AMADEUS_CLIENT_ID')
client_secret = os.environ.get('AMADEUS_CLIENT_SECRET')
OPENCAGE_API_KEY = os.environ.get('OPENCAGE_API_KEY')

country_data = pd.read_csv('country_info_extracted_data.csv')

# Helper functions

def get_amadeus_token(client_id, client_secret):
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    body = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    response = requests.post(url, headers=headers, data=body)
    
    if response.status_code == 200:
        token = response.json()['access_token']
        print("Successfully authenticated.")
        return token
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def convert_currency(amount, base_currency, target_currency, api_key):
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}/{amount}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['conversion_result']
    return 0.0

def get_capital_coordinates(country, city):
    geolocator = Nominatim(user_agent="SoloWanderlust")
    location = geolocator.geocode(f"{city}+{country}")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

def get_google_places_data(location, api_key, place_type="tourist attraction"):
    """Fetch places from Google Places API for a specific location and place type."""
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place_type}+in+{location}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['results']
    return []

def get_google_reviews(place_id, api_key):
    """Retrieve user reviews for a specific place using its place_id."""
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('result', {}).get('reviews', [])
    return []

def calculate_preference_match(user_preferences, google_data, api_key): # Calculate preference match using Cosine Similarity
    """
    Calculate preference match score using cosine similarity on reviews based on user preferences.
    User preferences contain keywords like 'adventure', 'culture', etc.
    """
    # Define user preferences as a single string for vectorization
    user_pref_text = " ".join(user_preferences.split()).lower()
    
    # Initialize total score and review count
    total_score = 0
    count = 0

    # Vectorizer to create keyword presence vectors for cosine similarity
    vectorizer = TfidfVectorizer().fit([user_pref_text])
    user_pref_vector = vectorizer.transform([user_pref_text]).toarray()

    for place in google_data:
        place_id = place.get('place_id')
        reviews = get_google_reviews(place_id, api_key)
        
        for review in reviews:
            review_text = review.get('text', '').lower()
            review_sentiment = TextBlob(review_text).sentiment.polarity  # Sentiment range from -1 to 1

            # Transform review into vector space using the same vectorizer
            review_vector = vectorizer.transform([review_text]).toarray()
            
            # Calculate cosine similarity between user preferences and review text
            similarity = cosine_similarity(user_pref_vector, review_vector)[0][0]
            
            # Weight similarity by sentiment to emphasize positive reviews
            weighted_similarity_score = similarity * (review_sentiment + 1)  # Scale sentiment

            # Accumulate the weighted similarity score
            if similarity > 0:  # Only consider relevant reviews with similarity > 0
                total_score += weighted_similarity_score * 100  # Scale up for clarity
                count += 1

    # Average and cap the preference match score to 0-100
    preference_match_score = (total_score / count) if count > 0 else 0
    return min(100, preference_match_score)

def calculate_safety_score(row): # Calculate Safety Score with Sentiment Analysis
    """Analyze safety sentiment and return a score based on safety description."""
    safety_text = row['Safety and Security']
    blob = TextBlob(safety_text)
    sentiment = blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)
    base_score = 100 + (sentiment * 50)  # Center at 100, modify by sentiment
    return max(0, min(100, base_score))

def get_amadeus_hotel_data(latitude, longitude, token):
    """Fetch hotel data from Amadeus API for a specific location."""
    url = f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-geocode?latitude={latitude}&longitude={longitude}&radius=10"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [hotel['hotelId'] for hotel in response.json()['data'][:20]]
    else:
        print(f"Error fetching hotel data: {response.status_code}, {response.text}")
        return []

def get_hotel_prices(hotel_ids, token):
    """Retrieve hotel prices for a list of hotels using their hotel_ids."""
    url = f"https://test.api.amadeus.com/v3/shopping/hotel-offers?hotelIds={hotel_ids}"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Extracting the base price for each hotel
        base_prices = []
        for hotel_offer in response.json()['data']:
            if 'offers' in hotel_offer:
                for offer in hotel_offer['offers']:
                    if 'base' in offer['price']:
                        base_price = offer['price']['base']  # Accessing the base price
                        base_prices.append(base_price)
        return base_prices
    else:
        print(f"Error fetching hotel prices: {response.status_code}, {response.text}")
        return []

def calculate_budget_fit(hotel_ids, user_budget, token):
    """Compare the user's budget with average hotel costs."""
    prices = get_hotel_prices(hotel_ids, token)

    budget_fit_score = 0
    if prices:
        avg_cost = sum(float(price) for price in prices) / len(prices)
        budget_fit_score = max(0, min(100, (user_budget / avg_cost) * 100))  # Normalize to 100
    return budget_fit_score

def calculate_social_score(places):
    """
    Calculate a social score based on Google Places ratings and user ratings count.
    The higher the rating and number of ratings, the higher the social score.
    """
    total_score = 0
    count = 0
    for place in places:
        rating = place.get('rating', 0)
        user_ratings_total = place.get('user_ratings_total', 0)

        # Score contribution from rating and popularity (user ratings count)
        place_social_score = rating * (user_ratings_total / 10)  # Scale by count

        # Normalize to cap each place's impact, then add to total
        total_score += min(place_social_score, 100)  # Cap score to prevent outliers
        count += 1

    # Normalize to 0-100 if places are found, else return 0
    return min(100, (total_score / count) if count > 0 else 0)

def rank_destinations(user_preferences, countries, coordinates, currencies, google_api_key, amadeus_api_key, weights): # Multi-Criteria Scoring
    """
    Rank destinations based on weighted criteria using user preferences.
    weights = (preference_weight, safety_weight, social_weight, budget_weight)
    """
    ranked_destinations = []
    for country, coordinate, currency in zip(countries, coordinates, currencies):
        row = country_data[country_data['Country'] == country].iloc[0]

        # Get data from Google Places for the country
        google_data = get_google_places_data(country, google_api_key)

        # Get data from Amadeus
        amadeus_data = get_amadeus_hotel_data(coordinate[0], coordinate[1], amadeus_api_key)
        
        # Calculate individual scores
        pref_match_score = calculate_preference_match(user_preferences["activities"], google_data, google_api_key)
        safety_score = calculate_safety_score(row)
        user_budget = convert_currency(user_preferences["budget"], 'USD', currency, EXCHANGE_RATE_API_KEY)
        budget_fit_score = calculate_budget_fit(amadeus_data, user_budget, amadeus_api_key)
        social_score = calculate_social_score(google_data)

        # Compute final score using the weighted criteria
        final_score = (
            weights[0] * pref_match_score +
            weights[1] * safety_score +
            weights[2] * social_score +
            weights[3] * budget_fit_score
        ) / sum(weights)  # Normalize by total weight

        ranked_destinations.append((country, final_score, {
            "Preference Match Score": pref_match_score,
            "Safety Score": safety_score,
            "Social Score": social_score,
            "Budget Fit Score": budget_fit_score
        }))

    # Sort by final score, descending
    ranked_destinations.sort(key=lambda x: x[1], reverse=True)
    return ranked_destinations

@app.route('/')
def index():
    countries = country_data['Country'].tolist()
    return render_template('index.html', countries=countries, OPENCAGE_API_KEY=OPENCAGE_API_KEY, cd=cd)

@app.route('/rank-destinations', methods=['POST'])
def rank_destinations_endpoint():
    data = request.get_json()
    user_preferences = data['user_preferences']
    countries = data['countries']
    coordinates = data['coordinates']
    currencies = data['currencies']
    weights = data['weights']

    # Get coordinates for each country’s capital
    # coordinates = [get_capital_coordinates(country, cd[country]) for country in countries]
    print(coordinates)

    # Fetch Amadeus token
    AMADEUS_TOKEN = get_amadeus_token(client_id, client_secret)
    if not AMADEUS_TOKEN:
        return jsonify({'error': 'Failed to get Amadeus token'}), 500

    # Call rank_destinations function
    ranked_results = rank_destinations(
        user_preferences,
        countries,
        coordinates,
        currencies,
        GOOGLE_PLACES_API_KEY,
        AMADEUS_TOKEN,
        weights
    )

    return jsonify(ranked_results)


if __name__ == "__main__":
    app.run(debug=True)