Creating a complete Python program for a project like "Smart-Waste-Manager" involves multiple components including IoT for data collection, a machine learning model for route optimization, and a web application for user interaction. The project requires integrating sensors to gather data, processing this data to predict optimal waste collection routes, and providing a user-friendly interface. Due to the multifaceted nature of such a project, I'll outline a basic version encompassing its major components using Python, Flask for web API, and Scikit-learn for machine learning. More comprehensive functionalities should be built upon this basic framework through collaborative development.

Here's a simplified version:

```python
# Import necessary libraries
from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import logging

# Create a Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Sample data: Stores the sensor data
# In a real-world scenario, this would be dynamic IoT data fetched in real-time
waste_bins_data = pd.DataFrame({
    'bin_id': [1, 2, 3],
    'latitude': [34.0522, 34.0523, 34.0524],
    'longitude': [-118.2437, -118.2436, -118.2435],
    'fill_level': [90, 65, 80]  # fill level percentage
})

# Route optimization (e.g., using KMeans for illustrative purposes)
def optimize_routes(data):
    try:
        # Use KMeans clustering to identify optimal routes (centroids)
        if len(data) < 2:
            raise ValueError("Insufficient data for clustering.")

        coordinates = data[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=2)  # Assuming two clusters/routes for simplicity
        clusters = kmeans.fit_predict(coordinates)
        data['cluster'] = clusters
        centroids = kmeans.cluster_centers_
        
        logging.info(f"Route optimization complete. Centroids: {centroids}")

        return centroids
    except Exception as e:
        logging.error(f"Error in route optimization: {e}")
        raise

# Load model (If there's a pre-trained model for predicting fill levels or other aspects, load it here)
# model = joblib.load('model.pkl')  # Example of loading a model

@app.route('/optimize', methods=['GET'])
def get_optimized_routes():
    try:
        centroids = optimize_routes(waste_bins_data)
        return jsonify({'routes': centroids.tolist()})
    except Exception as e:
        logging.error(f"Failed to get optimized routes: {e}")
        return jsonify({'error': 'Failed to optimize routes'}), 500

@app.route('/submit_sensor_data', methods=['POST'])
def submit_sensor_data():
    try:
        data = request.json
        # Validate data
        required_keys = {'bin_id', 'latitude', 'longitude', 'fill_level'}
        if not required_keys.issubset(data.keys()):
            raise ValueError("Missing keys in submitted data")

        # Here you would normally append to your database or in-memory data
        # For this example, we're just logging the received data
        logging.info(f"Received data: {data}")

        return jsonify({'status': 'success'})
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"Failed to submit sensor data: {e}")
        return jsonify({'error': 'Failed to submit sensor data'}), 500

# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
```

### Key Components:

1. **Flask**: A micro web framework for building the web-based application.
2. **KMeans Clustering**: A simple clustering algorithm to illustrate route optimization. In reality, advanced algorithms would consider traffic, waste levels, etc.
3. **Error Handling**: The program includes basic error handling to manage potential errors during data processing and API requests.
4. **Logging**: Used for tracking the application's flow and debugging.

### Considerations and Extensions:

- **IoT Integration**: Implement MQTT or WebSocket for real-time sensor data ingestion.
- **Advanced Machine Learning**: Replace the clustering with more advanced methods tailored to operational scenarios.
- **Database Integration**: Use a database (e.g., PostgreSQL, MongoDB) to store and manage waste bin data.
- **Frontend**: Develop a frontend interface with frameworks like React or Angular to interact with the Flask backend.
- **APIs for Admin and Reporting**: Implement user management and reporting tools for tracking efficiency and cost savings. 

For a real-world application, collaborating with stakeholders for requirements and continuous integration would be essential.