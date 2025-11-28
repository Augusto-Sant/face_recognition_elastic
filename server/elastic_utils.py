from elasticsearch import Elasticsearch
import config
from config import logger
import time


def get_elastic_client() -> Elasticsearch:
    try:
        time.sleep(15)
        es = Elasticsearch([f"http://{config.ES_HOST}:{config.ES_PORT}"])
        logger.info(
            f"ELASTIC | Connected to Elasticsearch at {config.ES_HOST}:{config.ES_PORT}"
        )
        return es
    except Exception as e:
        logger.error(f"ELASTIC | Failed to connect to Elasticsearch: {e}")
        raise e


def create_index():
    es = get_elastic_client()
    mapping = {
        "mappings": {
            "properties": {
                "person_id": {"type": "keyword"},
                "name": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": config.MODEL_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
                "timestamp": {"type": "date"},
            }
        }
    }
    try:
        if es.indices.exists(index=config.ES_INDEX):
            logger.info(f"ELSATIC | Index '{config.ES_INDEX}' already exists")
            return
        es.indices.create(index=config.ES_INDEX, body=mapping)
        logger.info(f"ELASTIC | Created index: {config.ES_INDEX}")
    except Exception as e:
        logger.error(f"ELASTIC | Failed to create index '{config.ES_INDEX}': {e}")
        raise e
