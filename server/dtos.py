from pydantic import BaseModel


class ResponseStoreFace(BaseModel):
    message: str
    person_id: str
    name: str
    es_id: str


class ResponseRecognizeFace(BaseModel):
    message: str
    best_match: dict
    all_matches: list
