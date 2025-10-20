const http = require('http');
const express = require('express');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(express.json({ limit: '50mb' })); // Increase payload limit for images

API_KEY = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Cropping control flags - set via API
let croppingMode = {
  NoCrop: true,      // Full image capture (default)
  CropLeft: false,   // Crop left player (213x480)
  CropMiddle: false, // Crop middle player (214x480)
  CropRight: false,  // Crop right player (213x480)
  CropCards: false   // Crop community cards (600x300 center)
};

// ROI definitions for each crop mode
const ROI_DEFINITIONS = {
  CropLeft: { x: 0, y: 0, width: 213, height: 480, section: 'LEFT' },
  CropMiddle: { x: 213, y: 0, width: 214, height: 480, section: 'MIDDLE' },
  CropRight: { x: 427, y: 0, width: 213, height: 480, section: 'RIGHT' },
  CropCards: { x: 20, y: 90, width: 600, height: 300, section: 'CARDS' } // Centered on 640x480
};

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

// Set cropping mode endpoint
app.post('/api/set-crop-mode', validateApiKey, (req, res) => {
  const { NoCrop, CropLeft, CropMiddle, CropRight, CropCards } = req.body;
  
  // Reset all flags to false first
  croppingMode = {
    NoCrop: false,
    CropLeft: false,
    CropMiddle: false,
    CropRight: false,
    CropCards: false
  };
  
  // Set the requested flags
  if (NoCrop !== undefined) croppingMode.NoCrop = NoCrop;
  if (CropLeft !== undefined) croppingMode.CropLeft = CropLeft;
  if (CropMiddle !== undefined) croppingMode.CropMiddle = CropMiddle;
  if (CropRight !== undefined) croppingMode.CropRight = CropRight;
  if (CropCards !== undefined) croppingMode.CropCards = CropCards;
  
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
    } else if (croppingMode.CropCards) {
      shouldCrop = true;
      targetROI = ROI_DEFINITIONS.CropCards;
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
    
    // Generate filename
    const filename = `${device_id || 'ESP32'}${roiSection}_${timestamp || Date.now()}.${image_format || 'jpg'}`;
    const filepath = path.join(uploadsDir, filename);
    
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
    cropMode: croppingMode
  });
});

// Start HTTP server
http.createServer(app).listen(3000, '0.0.0.0', () => {
  console.log('HTTP server running on port 3000');
  console.log('Endpoints:');
  console.log('  POST /api/signal - Send basic signals');
  console.log('  POST /api/upload-image - Upload camera images');
  console.log('  POST /api/set-crop-mode - Set cropping mode');
  console.log('  GET  /api/get-crop-mode - Get current cropping mode');
  console.log('  GET  /api/health - Health check');
  console.log(`Images will be saved to: ${uploadsDir}`);
  console.log('\n🔧 Default crop mode:', croppingMode);
});