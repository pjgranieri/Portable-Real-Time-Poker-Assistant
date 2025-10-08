const http = require('http');
const express = require('express');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(express.json({ limit: '50mb' })); // Increase payload limit for images

const API_KEY = process.env.API_KEY;

// Try to load sharp for image cropping
let sharp;
try {
  sharp = require('sharp');
  console.log('✅ Sharp loaded - image cropping enabled');
} catch(e) {
  console.log('⚠️  Sharp not installed - install with: npm install sharp');
}

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
app.post('/api/upload-image', validateApiKey, async (req, res) => {
  try {
    console.log('📸 Image upload request received');
    
    const { device_id, timestamp, image_format, image_width, image_height, image_size, image_data, roi } = req.body;
    
    // Validate required fields
    if (!image_data) {
      return res.status(400).json({ message: 'Missing image_data' });
    }

    // Decode base64 image
    const imageBuffer = Buffer.from(image_data, 'base64');
    
    let finalBuffer = imageBuffer;
    let finalWidth = image_width;
    let finalHeight = image_height;
    let roiSection = '';
    
    // If ROI metadata exists and sharp is available, crop the image
    if (roi && roi.x !== undefined && roi.width && roi.height && sharp) {
      roiSection = `_${roi.section}`;
      console.log(`🔲 Cropping ROI: ${roi.section} (x:${roi.x}, y:${roi.y}, ${roi.width}x${roi.height})`);
      
      try {
        finalBuffer = await sharp(imageBuffer)
          .extract({
            left: roi.x,
            top: roi.y,
            width: roi.width,
            height: roi.height
          })
          .jpeg({ quality: 85 })
          .toBuffer();
        
        finalWidth = roi.width;
        finalHeight = roi.height;
        
        console.log(`✂️  Cropped: ${finalBuffer.length} bytes (${finalWidth}x${finalHeight})`);
        console.log(`📉 Bandwidth saved: ${Math.round((1 - finalBuffer.length / imageBuffer.length) * 100)}%`);
      } catch (cropError) {
        console.error('❌ Cropping failed:', cropError.message);
        // Fall back to full image
      }
    } else if (roi && !sharp) {
      console.log('⚠️  ROI requested but Sharp not installed - saving full image');
      roiSection = `_${roi.section}_FULL`;
    }
    
    // Generate filename
    const filename = `${device_id || 'ESP32'}${roiSection}_${timestamp || Date.now()}.${image_format || 'jpg'}`;
    const filepath = path.join(uploadsDir, filename);
    
    // Save image to disk
    fs.writeFileSync(filepath, finalBuffer);
    
    console.log(`✅ Image saved: ${filename}`);
    console.log(`   Final size: ${finalBuffer.length} bytes (${finalWidth}x${finalHeight})`);
    console.log(`   Device: ${device_id}`);
    console.log(`   Path: ${filepath}\n`);
    
    res.json({ 
      message: 'Image uploaded successfully!',
      filename: filename,
      size: finalBuffer.length,
      dimensions: `${finalWidth}x${finalHeight}`,
      roi: roi || null,
      cropped: roi && sharp ? true : false
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