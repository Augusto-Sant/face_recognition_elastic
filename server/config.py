import logging
import os

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("face_api")

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "container_elastic")
ES_PORT = os.getenv("ELASTICSEARCH_PORT", "9200")
ES_INDEX = "face_embeddings"

MODEL_NAME = "Facenet"
MODEL_DIMENSIONS = 128
DETECTOR_BACKEND = "opencv"
