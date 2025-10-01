const express = require('express');
const multer = require('multer');
const { BlobServiceClient } = require('@azure/storage-blob');
const { CosmosClient } = require('@azure/cosmos');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Azure Storage configuration
const blobServiceClient = BlobServiceClient.fromConnectionString(
  process.env.AZURE_STORAGE_CONNECTION_STRING
);
const containerName = 'esp32-images';

// Azure Cosmos DB configuration (optional - for metadata)
const cosmosClient = new CosmosClient({
  endpoint: process.env.COSMOS_DB_ENDPOINT,
  key: process.env.COSMOS_DB_KEY,
});
const database = cosmosClient.database('esp32-data');
const container = database.container('image-metadata');

// API Key validation middleware (optional)
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

// Main image upload endpoint
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
    
    console.log(`📁 Uploading: ${filename} (${imageBuffer.length} bytes)`);

    // Upload to Azure Blob Storage
    const blockBlobClient = blobServiceClient
      .getContainerClient(containerName)
      .getBlockBlobClient(filename);

    const uploadOptions = {
      metadata: {
        deviceId: device_id,
        timestamp: (timestamp || Date.now()).toString(),
        originalSize: image_size?.toString(),
        width: image_width?.toString(),
        height: image_height?.toString(),
      }
    };

    await blockBlobClient.upload(imageBuffer, imageBuffer.length, uploadOptions);
    
    const blobUrl = blockBlobClient.url;
    console.log(`✅ Uploaded to: ${blobUrl}`);

    // Store metadata in Cosmos DB (optional)
    const metadata = {
      id: filename.replace('.', '_'), // Cosmos DB ID requirements
      deviceId: device_id,
      filename: filename,
      timestamp: timestamp || Date.now(),
      imageFormat: image_format || 'jpg',
      imageWidth: image_width,
      imageHeight: image_height,
      imageSize: image_size,
      blobUrl: blobUrl,
      uploadedAt: new Date().toISOString()
    };

    try {
      await container.items.create(metadata);
      console.log('💾 Metadata saved to Cosmos DB');
    } catch (cosmosError) {
      console.warn('⚠️ Failed to save metadata to Cosmos DB:', cosmosError.message);
      // Continue even if metadata save fails
    }

    // Response
    res.status(200).json({
      success: true,
      message: 'Image uploaded successfully',
      filename: filename,
      blobUrl: blobUrl,
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
    const limit = parseInt(req.query.limit) || 10;
    const deviceId = req.query.device_id;

    let query = 'SELECT * FROM c ORDER BY c.timestamp DESC';
    const parameters = [];

    if (deviceId) {
      query = 'SELECT * FROM c WHERE c.deviceId = @deviceId ORDER BY c.timestamp DESC';
      parameters.push({ name: '@deviceId', value: deviceId });
    }

    const { resources: items } = await container.items
      .query({ query, parameters })
      .fetchNext();

    res.json({
      success: true,
      count: items.length,
      images: items.slice(0, limit)
    });

  } catch (error) {
    console.error('❌ Query error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve images',
      message: error.message
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

app.listen(port, () => {
  console.log(`🚀 ESP32 Image Upload API running on port ${port}`);
  console.log(`📡 Health check: http://localhost:${port}/api/health`);
});

module.exports = app;