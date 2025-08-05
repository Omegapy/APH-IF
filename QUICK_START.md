# 🚀 APH-IF Quick Start Guide

**Version:** dev 0.0.1  
**Advanced Parallel HybridRAG - Intelligent Fusion**

---

## ⚡ 3-Step Launch

### 1. Prerequisites Check
```bash
# Verify Docker is installed
docker --version
docker-compose --version

# Check available ports
netstat -an | findstr "8000 8501 7474 7687"  # Windows
# netstat -an | grep "8000\|8501\|7474\|7687"  # Linux/Mac
```

### 2. Clone & Start
```bash
git clone https://github.com/Omegapy/APH-IF-Dev.git
cd APH-IF-Dev-v1
docker-compose up --build -d
```

### 3. Access Services
- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000/docs  
- **Neo4j Browser**: http://localhost:7474

---

## 🐳 Container Status

Check if all services are healthy:
```bash
docker-compose ps
```

Expected output:
```
NAME              STATUS
aph_if_backend    Up (healthy)
aph_if_frontend   Up (healthy)  
aph_if_neo4j      Up (healthy)
```

---

## 🔍 Quick Tests

### Test Backend Health
```bash
curl http://localhost:8000/health
```

### Test Processing Endpoint
```bash
curl -X POST "http://localhost:8000/generate_parallel_hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is APH-IF?", "max_results": 5}'
```

### Check Logs
```bash
# Backend logs
docker-compose logs aph_if_backend

# Frontend logs  
docker-compose logs aph_if_frontend

# All logs
docker-compose logs
```

---

## 🛠️ Common Commands

### Service Management
```bash
# Stop all services
docker-compose down

# Restart specific service
docker-compose restart aph_if_backend

# Rebuild and restart
docker-compose up --build -d

# View real-time logs
docker-compose logs -f aph_if_backend
```

### Troubleshooting
```bash
# Check container status
docker-compose ps

# Inspect specific container
docker inspect aph_if_backend

# Clean restart
docker-compose down
docker system prune -f
docker-compose up --build -d
```

---

## 🌐 Service Details

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:8501 | User interface & bot chat |
| **Backend API** | http://localhost:8000 | APH-IF processing engine |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |
| **Health Check** | http://localhost:8000/health | Service status |
| **Neo4j Browser** | http://localhost:7474 | Database interface |

---

## 🎯 What to Try

1. **Overview Interface** - Visit http://localhost:8501 for system overview
2. **Bot Chat** - Use the integrated bot interface for queries
3. **API Testing** - Explore http://localhost:8000/docs for API endpoints
4. **Health Monitoring** - Check service status and metrics
5. **Database Browser** - Explore the knowledge graph at http://localhost:7474

---

## ⚠️ Troubleshooting

### Port Conflicts
If ports are in use:
```bash
# Find processes using ports
netstat -ano | findstr "8000"  # Windows
# lsof -i :8000                # Linux/Mac

# Kill process if needed
taskkill /PID <PID> /F         # Windows
# kill -9 <PID>                # Linux/Mac
```

### Memory Issues
- Ensure **8GB+ RAM** available
- Close unnecessary applications
- Restart Docker Desktop

### Container Issues
```bash
# Clean restart
docker-compose down
docker system prune -f
docker-compose up --build -d
```

---

## 📚 Next Steps

1. **Read Full Documentation**: See [README.md](README.md)
2. **Explore API**: Visit http://localhost:8000/docs
3. **Try Bot Interface**: Chat at http://localhost:8501
4. **Check Configuration**: Review `docker-compose.yml`
5. **Development Setup**: See development section in README

---

**🎉 You're ready to explore Advanced Parallel HybridRAG!**

For detailed documentation, see [README.md](README.md)
