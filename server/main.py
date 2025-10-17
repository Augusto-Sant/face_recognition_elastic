from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from elasticsearch import Elasticsearch
from deepface import DeepFace
import numpy as np
import os
import io
from PIL import Image
import time
from pydantic import BaseModel
import logging

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("face_api")

app = FastAPI(title="Face Recognition API")

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "container_elastic")
ES_PORT = os.getenv("ELASTICSEARCH_PORT", "9200")
ES_INDEX = "face_embeddings"

try:
    es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])
    logger.info(f"Connected to Elasticsearch at {ES_HOST}:{ES_PORT}")
except Exception as e:
    logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
    raise e

MODEL_NAME = "Facenet"
MODEL_DIMENSIONS = 128
DETECTOR_BACKEND = "opencv"


def create_index():
    mapping = {
        "mappings": {
            "properties": {
                "person_id": {"type": "keyword"},
                "name": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": MODEL_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
                "timestamp": {"type": "date"},
            }
        }
    }
    try:
        if es.indices.exists(index=ES_INDEX):
            logger.info(f"Index '{ES_INDEX}' already exists")
            return
        es.indices.create(index=ES_INDEX, body=mapping)
        logger.info(f"Created index: {ES_INDEX}")
    except Exception as e:
        logger.error(f"❌ Failed to create index '{ES_INDEX}': {e}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Wait for Elasticsearch to be ready and create index"""
    max_retries = 30
    retry_delay = 5  # segundos entre tentativas

    for attempt in range(1, max_retries + 1):
        try:
            if es.ping():
                logger.info(f"[STARTUP] Elasticsearch is up on attempt {attempt}")
                create_index()
                return
            else:
                logger.warning(f"⚠️ [STARTUP] Elasticsearch not ready (attempt {attempt}/{max_retries})")
        except Exception as e:
            logger.warning(f"⚠️ [STARTUP] Connection failed (attempt {attempt}/{max_retries}): {e}")
        
        time.sleep(retry_delay)


    logger.error("❌ [STARTUP] Elasticsearch did not become ready in time. Exiting.")
    raise RuntimeError("Elasticsearch failed to start within expected time")


def image_to_array(image_file):
    image = Image.open(io.BytesIO(image_file))
    return np.array(image)


def get_face_embedding(image_array):
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
        if not embedding_objs:
            raise ValueError("No face detected in the image")
        return embedding_objs[0]["embedding"]
    except Exception as e:
        logger.error(f"❌ Face embedding failed: {e}")
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")


class ResponseStoreFace(BaseModel):
    message: str
    person_id: str
    name: str
    es_id: str


@app.post("/store")
async def store_face(
    file: UploadFile = File(...), person_id: str = Form(...), name: str = Form(...)
) -> ResponseStoreFace:
    start_total = time.time()
    logger.info(f"📥 Received request to store face: {person_id} ({name})")

    if not person_id or not name:
        raise HTTPException(status_code=400, detail="person_id and name are required")

    try:
        t0 = time.time()
        image_bytes = await file.read()
        image_array = image_to_array(image_bytes)

        embedding = get_face_embedding(image_array)

        t1 = time.time()
        doc = {
            "person_id": person_id,
            "name": name,
            "embedding": embedding,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        response = es.index(index=ES_INDEX, document=doc)

        total_time = time.time() - start_total
        logger.info(f"Face stored successfully in {total_time:.2f}s")

        return ResponseStoreFace(
            message="Face stored successfully",
            person_id=person_id,
            name=name,
            es_id=response["_id"],
        )
    except Exception as e:
        logger.error(f"❌ Error storing face: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing face: {str(e)}")


class ResponseRecognizeFace(BaseModel):
    message: str
    best_match: dict
    all_matches: list


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)) -> ResponseRecognizeFace:
    start_total = time.time()

    try:
        image_bytes = await file.read()
        image_array = image_to_array(image_bytes)

        embedding = get_face_embedding(image_array)

        knn_query = {
            "knn": {
                "field": "embedding",
                "query_vector": embedding,
                "k": 5,
                "num_candidates": 100,
            },
            "_source": ["person_id", "name", "timestamp"],
        }

        response = es.search(index=ES_INDEX, body=knn_query)

        if not response["hits"]["hits"]:
            logger.info("⚠️ No matching face found")
            return ResponseRecognizeFace(
                message="No matching face found",
                best_match={},
                all_matches=[],
            )

        matches = [
            {
                "person_id": hit["_source"]["person_id"],
                "name": hit["_source"]["name"],
                "similarity_score": hit["_score"],
                "timestamp": hit["_source"]["timestamp"],
            }
            for hit in response["hits"]["hits"]
            if hit["_score"] >= 0.70
        ]

        if not matches:
            logger.warning("⚠️ Matches found but all below threshold (0.70)")
            raise HTTPException(status_code=404, detail="No strong match found")

        total_time = time.time() - start_total
        logger.info(
            f"Recognition completed in {total_time:.2f}s — best match: {matches[0]['name']} ({matches[0]['similarity_score']:.2f})"
        )

        return ResponseRecognizeFace(
            message="Face recognition completed",
            best_match=matches[0],
            all_matches=matches,
        )

    except Exception as e:
        logger.error(f"❌ Error recognizing face: {e}")
        raise HTTPException(status_code=500, detail=f"Error recognizing face: {str(e)}")


@app.get("/health")
async def health_check():
    es_status = es.ping()
    logger.info(f"💓 Health check — Elasticsearch: {'OK' if es_status else 'DOWN'}")
    return {"status": "healthy", "elasticsearch": "connected" if es_status else "disconnected"}


@app.get("/")
async def root():
    return {
        "message": "Face Recognition API",
        "endpoints": {
            "/store": "POST - Store a face embedding",
            "/recognize": "POST - Recognize a face",
            "/health": "GET - Health check",
        },
    }
