const http = require('http');
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '50mb' })); // Increase payload limit for images

const API_KEY = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '../../Outputs');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// API key validation middleware
const validateApiKey = (req, res, next) => {
  const clientApiKey = req.headers['x-api-key'];
  if (clientApiKey !== API_KEY) {
    return res.status(403).json({ message: 'Forbidden: Invalid API Key' });
  }
  next();
};

// Basic signal endpoint
app.post('/api/signal', validateApiKey, (req, res) => {
  console.log('Signal received:', req.body);
  res.json({ message: 'Signal received successfully!' });
});

// Image upload endpoint
app.post('/api/upload-image', validateApiKey, (req, res) => {
  try {
    console.log('📸 Image upload request received');
    
    const { device_id, timestamp, image_format, image_width, image_height, image_size, image_data } = req.body;
    
    // Validate required fields
    if (!image_data) {
      return res.status(400).json({ message: 'Missing image_data' });
    }

    // Decode base64 image
    const imageBuffer = Buffer.from(image_data, 'base64');
    
    // Generate filename with timestamp
    const filename = `${device_id || 'ESP32'}_${timestamp || Date.now()}.${image_format || 'jpg'}`;
    const filepath = path.join(uploadsDir, filename);
    
    // Save image to disk
    fs.writeFileSync(filepath, imageBuffer);
    
    console.log(`✅ Image saved: ${filename}`);
    console.log(`   Size: ${image_size} bytes (${image_width}x${image_height})`);
    console.log(`   Device: ${device_id}`);
    console.log(`   Path: ${filepath}`);
    
    res.json({ 
      message: 'Image uploaded successfully!',
      filename: filename,
      size: image_size,
      dimensions: `${image_width}x${image_height}`
    });
    
  } catch (error) {
    console.error('❌ Error processing image upload:', error);
    res.status(500).json({ message: 'Internal server error', error: error.message });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: Date.now() });
});

// Start HTTP server
http.createServer(app).listen(3000, '0.0.0.0', () => {
  console.log('HTTP server running on port 3000');
  console.log('Endpoints:');
  console.log('  POST /api/signal - Send basic signals');
  console.log('  POST /api/upload-image - Upload camera images');
  console.log('  GET  /api/health - Health check');
  console.log(`Images will be saved to: ${uploadsDir}`);
});