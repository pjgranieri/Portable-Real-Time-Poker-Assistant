# ESP32 Camera Image Upload System

This system allows an ESP32 camera module to capture images and upload them to an Azure VM via HTTP API.

## Components

### 1. Server (Node.js)
**Location:** `Communication/Server/server.js`

The server provides three API endpoints:
- `POST /api/signal` - Basic signal/ping endpoint for testing connectivity
- `POST /api/upload-image` - Receive and save camera images
- `GET /api/health` - Health check endpoint

### 2. ESP32 Clients

#### ESP32-Signal-Comms.ino
**Location:** `Communication/ESP32-Signal-Comms.ino`

A simple client that sends ping/pong signals to test connectivity between ESP32 and the server.

#### ESP32-Camera-Upload.ino
**Location:** `Communication/ESP32-Camera-Upload.ino`

Full-featured camera client that:
- Initializes ESP32 camera module
- Captures images at regular intervals (default: 10 seconds)
- Encodes images to base64
- Uploads images to the server via HTTP POST
- Includes error handling and memory management

## Setup Instructions

### Server Setup (On Azure VM)

1. **Navigate to the server directory:**
   ```bash
   cd /home/azureuser/Computer-Vision-Powered-AI-Poker-Coach/Communication/Server
   ```

2. **Install dependencies:**
   ```bash
   npm install express
   ```

3. **Start the server:**
   ```bash
   node server.js
   ```

4. **Verify the server is running:**
   ```bash
   curl http://localhost:3000/api/health
   ```

5. **Ensure firewall allows port 3000:**
   ```bash
   sudo ufw allow 3000
   ```

### ESP32 Setup

#### For Testing Connectivity (ESP32-Signal-Comms.ino)

1. Open `Communication/ESP32-Signal-Comms.ino` in Arduino IDE
2. Update the following variables:
   - `ssid` - Your WiFi network name
   - `password` - Your WiFi password
   - `server` - Should be `"20.246.97.176"` (your VM IP, no http://)
3. Upload to your ESP32
4. Open Serial Monitor (115200 baud) to see connection logs

#### For Camera Image Upload (ESP32-Camera-Upload.ino)

1. Open `Communication/ESP32-Camera-Upload.ino` in Arduino IDE
2. Update the following variables:
   - `ssid` - Your WiFi network name
   - `password` - Your WiFi password
   - `serverURL` - Should be `"http://20.246.97.176:3000/api/upload-image"`
3. **IMPORTANT:** Verify camera pin configuration matches your ESP32 model
   - The file is configured for ESP32-S3 by default
   - If you have a different model, update the pin definitions
4. Upload to your ESP32
5. Open Serial Monitor (115200 baud) to see capture and upload logs

## Camera Pin Configuration

The default configuration in `ESP32-Camera-Upload.ino` is for **ESP32-S3**:

```cpp
#define XCLK_GPIO     15
#define SIOD_GPIO      4
#define SIOC_GPIO      5
#define VSYNC_GPIO     6
#define HREF_GPIO      7
#define PCLK_GPIO     13
// ... etc
```

If you have a different ESP32 camera model (AI-Thinker, WROVER, etc.), you need to update these pins. Common models are documented in the `CameraWebServerEx/camera_pins.h` file.

## Testing

### Test Basic Connectivity

```bash
curl -X POST http://20.246.97.176:3000/api/signal \
-H "Content-Type: application/json" \
-H "X-API-Key: ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g" \
-d '{"device_id": "test_device", "signal": "ping"}'
```

Expected response:
```json
{"message":"Signal received successfully!"}
```

### Test Image Upload (with test data)

You can test the image upload endpoint without an ESP32 by sending a small base64 encoded image.

## Uploaded Images

Images uploaded from the ESP32 will be saved to:
```
/home/azureuser/Computer-Vision-Powered-AI-Poker-Coach/Outputs/
```

Filenames follow the pattern:
```
ESP32_CAM_001_<timestamp>.jpg
```

## Configuration

### Server Configuration
- **Port:** 3000 (can be changed in `server.js`)
- **Max payload size:** 50MB (adjust in `express.json({ limit: '50mb' })`)
- **API Key:** `ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g`

### ESP32 Configuration
- **Capture interval:** 10 seconds (adjust `CAPTURE_INTERVAL` in code)
- **Image resolution:** VGA (640x480) (adjust `FRAMESIZE_VGA` in code)
- **JPEG quality:** 12 (0-63, lower = higher quality)
- **WiFi timeout:** 20 seconds
- **HTTP timeout:** 30 seconds

## Troubleshooting

### Server Issues

**Port already in use:**
```bash
# Check what's using port 3000
sudo lsof -i :3000

# Stop the process (if needed)
docker stop <container-id>
```

**Can't connect from ESP32:**
- Verify server is running: `curl http://localhost:3000/api/health`
- Check firewall: `sudo ufw status`
- Ensure server is bound to `0.0.0.0` (already configured)

### ESP32 Issues

**WiFi not connecting:**
- Double-check SSID and password
- Ensure ESP32 is within WiFi range
- Check Serial Monitor for error messages

**Camera initialization failed:**
- Verify pin configuration matches your hardware
- Ensure PSRAM is available (required for camera)
- Try reducing image resolution

**Upload fails:**
- Check server IP is correct
- Verify API key matches
- Ensure sufficient memory (check heap in Serial Monitor)
- Try reducing image quality or resolution

**Out of memory:**
- Reduce `FRAMESIZE_VGA` to `FRAMESIZE_QVGA`
- Increase `jpeg_quality` value (more compression)
- Increase capture interval

## API Reference

### POST /api/signal
Test connectivity with a simple ping.

**Headers:**
- `Content-Type: application/json`
- `X-API-Key: <your-api-key>`

**Body:**
```json
{
  "device_id": "ESP32_001",
  "signal": "ping"
}
```

### POST /api/upload-image
Upload a camera image.

**Headers:**
- `Content-Type: application/json`
- `X-API-Key: <your-api-key>`

**Body:**
```json
{
  "device_id": "ESP32_CAM_001",
  "timestamp": 123456789,
  "image_format": "jpg",
  "image_width": 640,
  "image_height": 480,
  "image_size": 12345,
  "image_data": "<base64-encoded-image>"
}
```

### GET /api/health
Check server status.

**Response:**
```json
{
  "status": "OK",
  "timestamp": 1234567890
}
```

## Security Notes

- The API key is currently hardcoded. For production, use environment variables.
- HTTP is used instead of HTTPS for simplicity. For production, implement HTTPS.
- The system is designed for use on a private WiFi network.

## Next Steps

1. Test basic connectivity with `ESP32-Signal-Comms.ino`
2. Once connectivity works, upload `ESP32-Camera-Upload.ino`
3. Monitor Serial Monitor for capture/upload status
4. Check `Outputs/` folder on VM for saved images
5. Adjust capture interval and image quality as needed
