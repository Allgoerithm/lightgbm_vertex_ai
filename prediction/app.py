import os
import json
from flask import Flask, jsonify, request
import lightgbm as lgb
import logging
import google.cloud.logging as cl
import google.cloud.storage as gs


def health_check_response():
    """We use the simplest implementation of a health check: it always responds with OK
    """
    return '200: OK'


def predict():
    """Function doing the actual prediction. The prediction is requested via POST, cf.
       https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/predict
    """
    try:
        input_data = request.get_json()
        instances = input_data['instances'] # get instances from POST data; parameters will be ignored
        predictions = model.predict(instances).tolist()   # generate our predictions for each instance

        # there may need to be extra steps to make sure the response is properly formatted
        response = jsonify({'predictions': predictions})
    except Exception as err:
        response = jsonify({'error': str(err)})
    return response


def create_app():
    """
    Create flask app.
    :return: flask app
    """

    # import environment variables set by Vertex AI
    aip_health_route = os.environ['AIP_HEALTH_ROUTE']  # route on which Vertex AI requests answers to health checks
    aip_predict_route = os.environ['AIP_PREDICT_ROUTE']  # route on which model predictions are requested

    app = Flask(__name__)
    app.add_url_rule(rule=aip_health_route, view_func=health_check_response)
    app.add_url_rule(rule=aip_predict_route, view_func=predict, methods=['POST'])
    return app


if __name__ == '__main__':
    BOOSTER_FILENAME = 'lightgbm_booster.txt'
    client = cl.Client()
    client.setup_logging()   # emit all log messages to Google Cloud Logging
    logging.info(f'starting app setup')

    # downloading booster
    aip_storage_uri = os.environ['AIP_STORAGE_URI']  # gs-path to directory with model artifacts
    logging.info(f'AIP_STORAGE_URI is {aip_storage_uri}')
    aip_storage_uri = aip_storage_uri.replace('gs://', '')
    if aip_storage_uri.endswith('/'):
        aip_storage_uri = aip_storage_uri[:-1]
    first_slash = aip_storage_uri.find('/')
    if first_slash > 0:
        bucket_name = aip_storage_uri[:first_slash] 
        booster_path = aip_storage_uri[first_slash+1:] + '/' + BOOSTER_FILENAME
    else:
        bucket_name = aip_storage_uri
        model_path = BOOSTER_FILENAME 
    logging.info(f'bucket_name: {bucket_name}, booster_path: {booster_path}')
    storage_client = gs.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(booster_path)
    blob.download_to_filename(BOOSTER_FILENAME)
    model = lgb.Booster(model_file=BOOSTER_FILENAME)  # this will be used as a global variable in the prediction function
    logging.info(f'Booster successfully downloaded and recreated in memory')

    app = create_app()
    aip_http_port = os.environ['AIP_HTTP_PORT']  # the port our app is expected to listen on (default is 8080)
    app.run(host='0.0.0.0', port=aip_http_port, debug=False)  # running on 0.0.0.0 is a Vertex AI requirement
