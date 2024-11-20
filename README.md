# SoloWanderlust
_Discover Destinations that Speak to You – Crafted for the Solo Journey._

SoloWanderlust is a travel destination finder targeted at solo travelers. It helps solo travelers discover destinations that align with their specific preferences, such as activities (e.g., adventure, cultural exploration), safety levels, budget, travel duration, and desired experiences (e.g., social interaction, relaxation). The tool will provide personalized destination recommendations, along with relevant information like safety ratings tailored for solo travelers.

## Demo Video

https://youtu.be/rbJLLJbMGa8

## Deployment

Web app hosted on Heroku: https://solowanderlust-80c5fd04c3e4.herokuapp.com/

## Installation

1. Clone the repository and navigate to the project directory
```shell
https://github.com/PearlynnT/CSCE470-project-checkpoint3.git
cd CSCE470-project-checkpoint3
```

2. Create a Python virtual environment
```shell
python -m venv env
```

3. Activate the virtual environment
```shell
source env/bin/activate
```

4. Install dependencies
```shell
pip install -r requirements.txt
```

5. Create a `.env` file and add your API Keys
```
GOOGLE_PLACES_API_KEY=your_google_places_api_key
AMADEUS_CLIENT_ID=your_amadeus_api_key
AMADEUS_CLIENT_SECRET=your_amadeus_api_secret
EXCHANGE_RATE_API_KEY=your_exchangerate_api_key
OPENCAGE_API_KEY=your_opencage_api_key
```