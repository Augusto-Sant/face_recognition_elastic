from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from deepface import DeepFace
import numpy as np
import os
import io
from PIL import Image
import base64
import time
from pydantic import BaseModel
import logging

logger = logging.getLogger("uvicorn")

app = FastAPI(title="Face Recognition API")

# Elasticsearch configuration
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "container_elastic")
ES_PORT = os.getenv("ELASTICSEARCH_PORT", "9200")
ES_INDEX = "face_embeddings"

# Initialize Elasticsearch client
try:
    es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])
except Exception as e:
    raise Exception(f"nao conseguiu conectar com elastic aqui {e}")

# Model configuration
MODEL_NAME = "Facenet"
MODEL_DIMENSIONS = 128
DETECTOR_BACKEND = "opencv"


def create_index():
    """Create Elasticsearch index with proper mapping for KNN search"""
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
            print(f"Index already exists: {ES_INDEX}")
            return
        es.indices.create(index=ES_INDEX, body=mapping)
        print(f"Created index: {ES_INDEX}")
    except Exception as e:
        raise Exception(f"nao conseguiu criar elastic index: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize Elasticsearch connection and create index"""
    time.sleep(30)
    create_index()


def image_to_array(image_file):
    """Convert uploaded image to numpy array"""
    image = Image.open(io.BytesIO(image_file))
    return np.array(image)


def get_face_embedding(image_array):
    """Extract face embedding using DeepFace"""
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )

        if not embedding_objs:
            raise ValueError("No face detected in the image")

        # Return the first face embedding
        return embedding_objs[0]["embedding"]

    except Exception as e:
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
    """
    Store a face embedding in Elasticsearch

    Args:
        file: Image file containing a face
        person_id: Unique identifier for the person
        name: Name of the person
    """
    logger.info(f"Storing face for {person_id} with name {name}")
    if not person_id or not name:
        raise HTTPException(status_code=400, detail="person_id and name are required")

    try:
        # Read image
        image_bytes = await file.read()
        image_array = image_to_array(image_bytes)

        # Get face embedding
        embedding = get_face_embedding(image_array)

        # Store in Elasticsearch
        doc = {
            "person_id": person_id,
            "name": name,
            "embedding": embedding,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        response = es.index(index=ES_INDEX, document=doc)

        return ResponseStoreFace(
            status_code=200,
            message="Face stored successfully",
            person_id=person_id,
            name=name,
            es_id=response["_id"],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing face: {str(e)}")


class ResponseRecognizeFace(BaseModel):
    message: str
    best_match: dict
    all_matches: list


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)) -> ResponseRecognizeFace:
    """
    Recognize a face by finding the most similar embedding using KNN search

    Args:
        file: Image file containing a face to recognize
    """
    try:
        # Read image
        image_bytes = await file.read()
        image_array = image_to_array(image_bytes)

        # Get face embedding
        embedding = get_face_embedding(image_array)

        # Perform KNN search in Elasticsearch
        knn_query = {
            "knn": {
                "field": "embedding",
                "query_vector": embedding,
                "k": 5,  # Return top 5 matches
                "num_candidates": 100,
            },
            "_source": ["person_id", "name", "timestamp"],
        }

        response = es.search(index=ES_INDEX, body=knn_query)

        if not response["hits"]["hits"]:
            return ResponseRecognizeFace(
                status_code=200,
                message="No matching face found",
                best_match={},
                all_matches=[],
            )

        # Format results
        matches = []
        for hit in response["hits"]["hits"]:
            matches.append(
                {
                    "person_id": hit["_source"]["person_id"],
                    "name": hit["_source"]["name"],
                    "similarity_score": hit["_score"],
                    "timestamp": hit["_source"]["timestamp"],
                }
            )

        return ResponseRecognizeFace(
            status_code=200,
            message="Face recognition completed",
            best_match=matches[0],
            all_matches=matches,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recognizing face: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    es_status = es.ping()
    return {
        "status": "healthy",
        "elasticsearch": "connected" if es_status else "disconnected",
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "endpoints": {
            "/store": "POST - Store a face embedding",
            "/recognize": "POST - Recognize a face",
            "/health": "GET - Health check",
        },
    }
