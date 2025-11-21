# Deployment Guide - Modular Semantic Discovery System

**Agent 10: Documentation & Deployment Manager**
**Version:** 1.0.0

---

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt

# Start API server
python -m midi_generator.api.server

# Access API
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and start services
cd midi_generator/deployment
chmod +x scripts/deploy.sh
./scripts/deploy.sh production deploy

# Check status
docker-compose -f docker/docker-compose.yml ps

# View logs
docker-compose -f docker/docker-compose.yml logs -f api

# Stop services
./scripts/deploy.sh production stop
```

---

## Directory Structure

```
deployment/
├── docker/
│   ├── Dockerfile              # API container definition
│   ├── docker-compose.yml      # Multi-service orchestration
│   └── nginx.conf              # Reverse proxy configuration
├── scripts/
│   ├── deploy.sh               # Main deployment script
│   ├── backup.sh               # Backup script
│   └── monitor.sh              # Monitoring script
├── kubernetes/                 # K8s manifests (future)
└── README.md                   # This file
```

---

## Deployment Options

### 1. Docker (Recommended)

**Advantages:**
- Isolated environment
- Easy scaling
- Consistent across machines
- Production-ready

**Usage:**
```bash
# Quick deploy
./scripts/deploy.sh production deploy

# Custom configuration
DOCKER_REGISTRY=myregistry.com \
VERSION=1.0.0 \
./scripts/deploy.sh production deploy
```

### 2. Kubernetes (Production at Scale)

```bash
# Apply manifests
kubectl apply -f kubernetes/

# Check status
kubectl get pods -n midi-dna

# Scale
kubectl scale deployment midi-dna-api --replicas=5
```

### 3. Serverless (Cloud Functions)

```bash
# Deploy to AWS Lambda
serverless deploy --stage production

# Deploy to Google Cloud Functions
gcloud functions deploy midi-dna-api \
  --runtime python39 \
  --trigger-http
```

---

## Configuration

### Environment Variables

```bash
# API Configuration
export MODEL_PATH=/app/models
export LOG_LEVEL=info
export API_PORT=8000
export WORKERS=4

# CUDA/GPU
export CUDA_VISIBLE_DEVICES=0

# Redis (caching)
export REDIS_URL=redis://localhost:6379

# Model optimization
export USE_QUANTIZATION=true
export BATCH_SIZE=32
```

### Pre-trained Models

Download pre-trained models:

```bash
# Option 1: Direct download
wget https://example.com/models/modular_semantic_discovery_v1.pth \
  -O models/modular_semantic_discovery_v1.pth

# Option 2: Train from scratch
python -m midi_generator.learning.train_modular_pipeline \
  --corpus data/midi_corpus \
  --output models \
  --features 120 \
  --epochs 100 \
  --device cuda
```

---

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Response:
# {
#   "status": "healthy",
#   "pipeline_loaded": true,
#   "version": "1.0.0"
# }
```

### Metrics

```bash
# View statistics
curl http://localhost:8000/stats

# Prometheus metrics (if enabled)
curl http://localhost:8000/metrics
```

### Logging

```bash
# Docker logs
docker-compose logs -f api

# Log levels
export LOG_LEVEL=debug  # debug, info, warning, error
```

---

## Performance Optimization

### 1. Model Quantization

Reduces model size by 4x and inference time by 4x:

```python
pipeline = ModularSemanticDiscoveryPipeline.load_pretrained(
    quantize=True
)
```

### 2. Batch Processing

Process multiple files in parallel:

```python
dna_batch = pipeline.extract_dna_batch([
    "song1.mid",
    "song2.mid",
    "song3.mid"
], batch_size=32)
```

### 3. Caching

Use Redis for caching frequent requests:

```python
# Enable Redis caching
export REDIS_URL=redis://localhost:6379

# Cache TTL
export CACHE_TTL=3600  # 1 hour
```

### 4. GPU Acceleration

```bash
# Use NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Security

### 1. API Authentication

```python
# Add API key authentication
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/extract_dna")
async def extract_dna(
    file: UploadFile,
    api_key: str = Security(api_key_header)
):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of handler
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/extract_dna")
@limiter.limit("10/minute")
async def extract_dna(...):
    # ... handler
```

### 3. CORS Configuration

```python
# Configure allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## Troubleshooting

### Issue: Container won't start

```bash
# Check logs
docker-compose logs api

# Check for port conflicts
lsof -i :8000

# Rebuild image
docker-compose build --no-cache api
```

### Issue: Out of memory

```bash
# Reduce batch size
export BATCH_SIZE=16

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Increase Docker memory limit
docker-compose.yml:
  services:
    api:
      deploy:
        resources:
          limits:
            memory: 16G  # Increase from 8G
```

### Issue: Slow inference

```bash
# Enable model quantization
export USE_QUANTIZATION=true

# Use GPU
export CUDA_VISIBLE_DEVICES=0

# Enable caching
export REDIS_URL=redis://localhost:6379
```

---

## Backup and Recovery

### Backup Models

```bash
# Backup pre-trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup to S3
aws s3 cp models/ s3://your-bucket/models/ --recursive
```

### Backup Database (if using)

```bash
# PostgreSQL
pg_dump midi_dna > backup.sql

# MongoDB
mongodump --db midi_dna --out backup/
```

---

## Scaling

### Horizontal Scaling (Multiple Instances)

```bash
# Docker Compose
docker-compose up --scale api=5

# Kubernetes
kubectl scale deployment midi-dna-api --replicas=10
```

### Load Balancing

```nginx
# nginx.conf
upstream api_backend {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    location / {
        proxy_pass http://api_backend;
    }
}
```

---

## Production Checklist

- [ ] Pre-trained models downloaded and verified
- [ ] Environment variables configured
- [ ] Docker images built and tested
- [ ] Health checks passing
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Logging configured
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] SSL/TLS certificates installed
- [ ] Load balancing configured (if scaling)
- [ ] Documentation updated
- [ ] Team trained on deployment process

---

## Support

For issues or questions:

- GitHub Issues: https://github.com/doseedo/Do/issues
- Documentation: `/midi_generator/docs/MODULAR_SEMANTIC_DISCOVERY.md`
- API Docs: http://localhost:8000/docs

---

**Last Updated:** November 21, 2025
**Maintainer:** Agent 10 - Documentation & Deployment Manager
