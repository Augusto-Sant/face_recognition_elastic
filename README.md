# Face Recognition API

API de reconhecimento facial usando DeepFace e Elasticsearch para armazenar e identificar faces.

## Como Funciona

1. **Store**: Extrai embedding facial e armazena no Elasticsearch
2. **Recognize**: Compara face com banco usando busca KNN (threshold: 0.70)
3. **List**: Lista faces armazenadas com paginação

### Reconhecimento Facial

O DeepFace converte cada rosto em um vetor numérico (embedding) de 512 dimensões que representa características únicas da face. No reconhecimento, o Elasticsearch usa busca KNN (K-Nearest Neighbors) para encontrar os 5 embeddings mais similares no banco, calculando a distância vetorial. Apenas matches com score ≥ 0.70 são retornados como correspondências válidas.

## Tecnologias

- FastAPI + DeepFace (extração de embeddings)
- Elasticsearch 8.11 (busca vetorial KNN)
- Docker Compose

## Como Executar

```bash
# Subir serviços
docker-compose up -d

# API disponível em http://localhost:8000
# Elasticsearch em http://localhost:9200
```

## Endpoints

- `POST /store` - Armazena face (params: `file`, `person_id`, `name`)
- `POST /recognize` - Reconhece face (param: `file`)
- `GET /faces?page=1&size=10` - Lista faces
- `GET /health` - Status da API

## Exemplo de Uso

```python
import requests

# Armazenar face
with open("foto.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/store",
        files={"file": f},
        data={"person_id": "123", "name": "João Silva"}
    )
    print(response.json())

# Reconhecer face
with open("foto_teste.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/recognize",
        files={"file": f}
    )
    print(response.json())
```


## Estrutura

```
.
├── docker-compose.yaml
└── server/
    ├── main.py           # Endpoints da API
    ├── config.py         # Configurações
    ├── elastic_utils.py  # Funções do Elasticsearch
    ├── dtos.py           # Modelos de resposta
    └── Dockerfile
```
