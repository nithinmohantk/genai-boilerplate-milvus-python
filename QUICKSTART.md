# 🚀 Quick Start Guide

Get your GenAI RAG system running in 5 minutes!

## 1. One-Command Setup

```bash
# Clone and setup everything
git clone https://github.com/yourusername/genai-boilerplate-milvus-python.git
cd genai-boilerplate-milvus-python
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## 2. Quick Test

```bash
# Test the API
curl http://localhost:8000/api/v1/health

# Upload a document
echo "Python is a programming language" > test.txt
curl -X POST -F "file=@test.txt" http://localhost:8000/api/v1/documents/upload

# Search documents
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "programming", "top_k": 3}' \
  http://localhost:8000/api/v1/documents/search
```

## 3. Access Web Interfaces

- **API Docs**: http://localhost:8000/docs
- **Milvus Admin**: http://localhost:3001
- **MinIO Console**: http://localhost:9001

## 4. Python Example

```python
import asyncio
import httpx

async def quick_test():
    async with httpx.AsyncClient() as client:
        # Health check
        health = await client.get("http://localhost:8000/api/v1/health")
        print("Health:", health.json()["status"])
        
        # Upload document
        files = {"file": ("test.txt", "AI is transforming the world")}
        upload = await client.post("http://localhost:8000/api/v1/documents/upload", files=files)
        print("Upload:", upload.json()["message"])
        
        # Search
        search = await client.post("http://localhost:8000/api/v1/documents/search", 
                                 json={"query": "AI transformation", "top_k": 1})
        print("Search results:", len(search.json()["results"]))

asyncio.run(quick_test())
```

## 5. Next Steps

1. Add your OpenAI API key to `backend/.env`
2. Try the question-answering endpoint
3. Upload PDF/DOCX files
4. Explore the comprehensive examples in `backend/examples/`

## Troubleshooting

```bash
# Check services
docker-compose ps

# View logs
docker-compose logs api
docker-compose logs milvus

# Restart services
docker-compose restart

# Stop everything
docker-compose down
```

That's it! Your RAG system is ready to use! 🎉
