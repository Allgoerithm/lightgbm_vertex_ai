import json
from google.cloud import aiplatform

aiplatform.init(project='bt-pp-dsc-1th4', location='europe-west1')

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/871391074384/locations/europe-west1/endpoints/5276776204022579200" 
)

# A test example we'll send to our model for prediction
instances = json.dumps({'instances': [[8, 304, 150, 3433, 12, 70, 1]]})

response = endpoint.predict([[8, 304, 150, 3433, 12, 70, 1]])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0])