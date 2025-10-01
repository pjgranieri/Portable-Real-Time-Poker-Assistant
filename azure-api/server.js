const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;
const httpsPort = process.env.HTTPS_PORT || 3443;

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// API Key validation middleware
const validateApiKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  const validApiKey = process.env.API_KEY;
  
  if (validApiKey && apiKey !== validApiKey) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  next();
};

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'ESP32 Image Upload API'
  });
});

// Main image upload endpoint - simplified without Azure
app.post('/api/upload-image', validateApiKey, async (req, res) => {
  try {
    console.log('📤 Received image upload request');
    
    const {
      device_id,
      timestamp,
      image_format,
      image_width,
      image_height,
      image_size,
      image_data
    } = req.body;

    // Validate required fields
    if (!device_id || !image_data) {
      return res.status(400).json({ 
        error: 'Missing required fields: device_id, image_data' 
      });
    }

    // Decode base64 image
    const imageBuffer = Buffer.from(image_data, 'base64');
    
    // Generate unique filename
    const filename = `${device_id}_${timestamp || Date.now()}.${image_format || 'jpg'}`;
    const filepath = path.join(uploadsDir, filename);
    
    console.log(`📁 Saving: ${filename} (${imageBuffer.length} bytes)`);

    // Save to local filesystem
    fs.writeFileSync(filepath, imageBuffer);
    
    console.log(`✅ Saved to: ${filepath}`);

    // Response
    res.status(200).json({
      success: true,
      message: 'Image uploaded successfully',
      filename: filename,
      filepath: filepath,
      metadata: {
        size: imageBuffer.length,
        dimensions: `${image_width}x${image_height}`,
        format: image_format
      }
    });

  } catch (error) {
    console.error('❌ Upload error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Get recent images endpoint
app.get('/api/images', validateApiKey, async (req, res) => {
  try {
    const files = fs.readdirSync(uploadsDir)
      .filter(file => file.match(/\.(jpg|jpeg|png)$/i))
      .map(file => {
        const stats = fs.statSync(path.join(uploadsDir, file));
        return {
          filename: file,
          size: stats.size,
          created: stats.birthtime
        };
      })
      .sort((a, b) => b.created - a.created)
      .slice(0, 10);

    res.json({
      success: true,
      count: files.length,
      images: files
    });

  } catch (error) {
    console.error('❌ Query error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve images'
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('💥 Unhandled error:', error);
  res.status(500).json({
    success: false,
    error: 'Internal server error'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

// HTTPS configuration
let httpsOptions;
try {
  httpsOptions = {
    key: fs.readFileSync('./certs/key.pem'),
    cert: fs.readFileSync('./certs/cert.pem')
  };
  console.log('📋 Using development SSL certificates');
} catch (certError) {
  console.log('⚠️  No SSL certificates found, running HTTP only');
  httpsOptions = null;
}

// Start servers
if (httpsOptions) {
  https.createServer(httpsOptions, app).listen(httpsPort, '0.0.0.0', () => {
    console.log(`🔒 HTTPS ESP32 Image Upload API running on port ${httpsPort}`);
  });
}

app.listen(port, '0.0.0.0', () => {
  console.log(`🚀 HTTP ESP32 Image Upload API running on port ${port}`);
});