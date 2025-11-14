const http = require('http');
const express = require('express');
const fs = require('fs');
const path = require('path');
const os = require('os');

const app = express();
app.use(express.json({ limit: '50mb' })); // Increase payload limit for images

const API_KEY = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Configure save directory - UPDATE THIS to your desired path
const SAVE_DIRECTORY = path.join(os.homedir(), 'test_outputs_locally');

// Cropping control flags - set via API
let croppingMode = {
  NoCrop: true,      // Full image capture (default)
  CropLeft: false,   // Crop left section
  CropMiddle: false, // Crop middle section
  CropRight: false,  // Crop right section
};

// ROI definitions for HD (1280x720) images
const ROI_DEFINITIONS = {
  CropLeft: { x: 0, y: 0, width: 427, height: 720, section: 'LEFT' },
  CropMiddle: { x: 427, y: 0, width: 427, height: 720, section: 'MIDDLE' },
  CropRight: { x: 854, y: 0, width: 426, height: 720, section: 'RIGHT' },
};

// Try to load sharp for image cropping
let sharp;
try {
  sharp = require('sharp');
  console.log('✅ Sharp loaded - image cropping enabled');
} catch(e) {
  console.log('⚠️  Sharp not installed - install with: npm install sharp');
  console.log('   Server will still work, but cropping will be disabled');
}

// Create save directory if it doesn't exist
if (!fs.existsSync(SAVE_DIRECTORY)) {
  fs.mkdirSync(SAVE_DIRECTORY, { recursive: true });
  console.log(`📁 Created save directory: ${SAVE_DIRECTORY}`);
}

// API key validation middleware
const validateApiKey = (req, res, next) => {
  const clientApiKey = req.headers['x-api-key'];
  if (clientApiKey !== API_KEY) {
    return res.status(403).json({ message: 'Forbidden: Invalid API Key' });
  }
  next();
};

// Set cropping mode endpoint
app.post('/api/set-crop-mode', validateApiKey, (req, res) => {
  const { NoCrop, CropLeft, CropMiddle, CropRight } = req.body;
  
  // Reset all flags to false first
  croppingMode = {
    NoCrop: false,
    CropLeft: false,
    CropMiddle: false,
    CropRight: false,
  };
  
  // Set the requested flags
  if (NoCrop !== undefined) croppingMode.NoCrop = NoCrop;
  if (CropLeft !== undefined) croppingMode.CropLeft = CropLeft;
  if (CropMiddle !== undefined) croppingMode.CropMiddle = CropMiddle;
  if (CropRight !== undefined) croppingMode.CropRight = CropRight;
  
  console.log('🔧 Cropping mode updated:', croppingMode);
  
  res.json({ 
    message: 'Cropping mode updated successfully!',
    mode: croppingMode
  });
});

// Get current cropping mode endpoint
app.get('/api/get-crop-mode', validateApiKey, (req, res) => {
  res.json({ mode: croppingMode });
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
    let shouldCrop = false;
    let targetROI = null;
    
    // Determine which ROI to use based on cropping mode
    if (croppingMode.NoCrop) {
      console.log('📷 NoCrop mode - saving full image');
      roiSection = '';
    } else if (croppingMode.CropLeft) {
      shouldCrop = true;
      targetROI = ROI_DEFINITIONS.CropLeft;
    } else if (croppingMode.CropMiddle) {
      shouldCrop = true;
      targetROI = ROI_DEFINITIONS.CropMiddle;
    } else if (croppingMode.CropRight) {
      shouldCrop = true;
      targetROI = ROI_DEFINITIONS.CropRight;
    }
    
    // Perform cropping if required and sharp is available
    if (shouldCrop && targetROI && sharp) {
      roiSection = `_${targetROI.section}`;
      console.log(`🔲 Cropping ROI: ${targetROI.section} (x:${targetROI.x}, y:${targetROI.y}, ${targetROI.width}x${targetROI.height})`);
      
      try {
        finalBuffer = await sharp(imageBuffer)
          .extract({
            left: targetROI.x,
            top: targetROI.y,
            width: targetROI.width,
            height: targetROI.height
          })
          .jpeg({ quality: 85 })
          .toBuffer();
        
        finalWidth = targetROI.width;
        finalHeight = targetROI.height;
        
        console.log(`✂️  Cropped: ${finalBuffer.length} bytes (${finalWidth}x${finalHeight})`);
        console.log(`📉 Bandwidth saved: ${Math.round((1 - finalBuffer.length / imageBuffer.length) * 100)}%`);
      } catch (cropError) {
        console.error('❌ Cropping failed:', cropError.message);
        roiSection = '_FULL';
        // Fall back to full image
      }
    } else if (shouldCrop && !sharp) {
      console.log('⚠️  Cropping requested but Sharp not installed - saving full image');
      roiSection = `_${targetROI ? targetROI.section : 'UNKNOWN'}_FULL`;
    }
    
    // Generate filename with timestamp
    const now = new Date();
    const dateStr = now.toISOString().replace(/[:.]/g, '-').replace('T', '_').split('.')[0];
    const filename = `${device_id || 'ESP32'}${roiSection}_${dateStr}.${image_format || 'jpg'}`;
    const filepath = path.join(SAVE_DIRECTORY, filename);
    
    // Save image to disk
    fs.writeFileSync(filepath, finalBuffer);
    
    console.log(`✅ Image saved: ${filename}`);
    console.log(`   Final size: ${finalBuffer.length} bytes (${finalWidth}x${finalHeight})`);
    console.log(`   Device: ${device_id}`);
    console.log(`   Crop mode:`, Object.keys(croppingMode).find(key => croppingMode[key]));
    console.log(`   Path: ${filepath}\n`);
    
    res.json({ 
      message: 'Image uploaded successfully!',
      filename: filename,
      filepath: filepath,
      size: finalBuffer.length,
      dimensions: `${finalWidth}x${finalHeight}`,
      roi: targetROI || null,
      cropped: shouldCrop && sharp ? true : false,
      cropMode: Object.keys(croppingMode).find(key => croppingMode[key])
    });
    
  } catch (error) {
    console.error('❌ Error processing image upload:', error);
    res.status(500).json({ message: 'Internal server error', error: error.message });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: Date.now(),
    cropMode: croppingMode,
    saveDirectory: SAVE_DIRECTORY
  });
});

// Get local IP address
function getLocalIP() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return 'localhost';
}

// Start HTTP server
const PORT = 3000;
http.createServer(app).listen(PORT, '0.0.0.0', () => {
  const localIP = getLocalIP();
  
  console.log('=' .repeat(60));
  console.log('ESP32-CAM Local Image Server (Node.js)');
  console.log('=' .repeat(60));
  console.log('🌐 Server running on:');
  console.log(`   Local:   http://localhost:${PORT}/api/upload-image`);
  console.log(`   Network: http://${localIP}:${PORT}/api/upload-image`);
  console.log('');
  console.log('📁 Save directory:', SAVE_DIRECTORY);
  console.log('🔑 API Key:', API_KEY);
  console.log('');
  console.log('Available Endpoints:');
  console.log('  POST /api/upload-image - Upload camera images');
  console.log('  POST /api/set-crop-mode - Set cropping mode');
  console.log('  GET  /api/get-crop-mode - Get current cropping mode');
  console.log('  GET  /api/health - Health check');
  console.log('');
  console.log('🔧 Default crop mode:', croppingMode);
  console.log('');
  console.log('Update your ESP32 code with the Network URL above');
  console.log('Press Ctrl+C to stop the server');
  console.log('=' .repeat(60));
});