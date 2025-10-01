# Docker Setup for ESP32 Image API

This guide explains how to run the ESP32 Image Upload API using Docker.

## Prerequisites

- Docker and Docker Compose installed
- Azure Storage Account (update `.env` with your connection string)
- Your VM's public IP address

## Quick Start

### 1. Update Environment Variables

Edit `.env` file with your actual Azure credentials:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=youraccount;AccountKey=yourkey;EndpointSuffix=core.windows.net

# Cosmos DB (optional)
COSMOS_DB_ENDPOINT=https://youraccount.documents.azure.com:443/
COSMOS_DB_KEY=your_cosmos_key

# API Security
API_KEY=your_secure_random_api_key

# Server ports
PORT=3000
HTTPS_PORT=3443
```

### 2. Build and Run (Development)

```bash
# Build and start the containers
docker-compose up -d

# View logs
docker-compose logs -f esp32-api

# Stop containers
docker-compose down
```

### 3. Build and Run (Development with hot reload)

```bash
# Use development compose file
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f esp32-api-dev
```

### 4. Production with Nginx (Optional)

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d

# This will start both the API and nginx on ports 80/443
```

## API Endpoints

Once running, your API will be available at:

- **HTTP**: `http://your-vm-ip:3000`
- **HTTPS**: `https://your-vm-ip:3443`

### Health Check
- `GET /api/health` - Returns API status

### Image Upload
- `POST /api/upload-image` - Upload image from ESP32
  - Requires `X-API-Key` header
  - Accepts JSON with base64 image data

### Get Images
- `GET /api/images` - List recent uploaded images

## ESP32 Configuration

Update your ESP32 code with:

```cpp
const char* serverURL = "https://YOUR_VM_PUBLIC_IP:3443/api/upload-image";
const char* apiKey = "your_secure_random_api_key"; // Same as in .env
```

## SSL Certificates

### Development
Self-signed certificates are automatically generated in `./certs/`

### Production
For production, replace the certificates in `./certs/` or use Let's Encrypt:

```bash
# Get your VM's public IP
curl ifconfig.me

# Install certbot
sudo apt update && sudo apt install certbot

# Get certificates (replace with your domain)
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates to project
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./certs/key.pem
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./certs/cert.pem
sudo chown $USER:$USER ./certs/*.pem
```

## Troubleshooting

### Check container status
```bash
docker-compose ps
```

### View logs
```bash
docker-compose logs esp32-api
```

### Access container shell
```bash
docker-compose exec esp32-api sh
```

### Rebuild containers
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Test API manually
```bash
# Test health check
curl http://localhost:3000/api/health

# Test HTTPS
curl -k https://localhost:3443/api/health
```

## File Structure

```
Computer-Vision-Powered-AI-Poker-Coach/
├── Dockerfile                 # Main container definition
├── docker-compose.yml         # Production orchestration
├── docker-compose.dev.yml     # Development orchestration
├── .dockerignore              # Files to exclude from build
├── nginx.conf                 # Reverse proxy config
├── .env                       # Environment variables
├── certs/                     # SSL certificates
│   ├── cert.pem
│   └── key.pem
├── logs/                      # Application logs
├── azure-api/                 # Node.js application
│   ├── server.js
│   ├── package.json
│   └── healthcheck.js
└── DOCKER_README.md           # This file
```

## Security Notes

- Change the default API key in `.env`
- Use proper SSL certificates in production
- Consider using environment-specific `.env` files
- Regularly update container images
- Monitor logs in `./logs/` directory