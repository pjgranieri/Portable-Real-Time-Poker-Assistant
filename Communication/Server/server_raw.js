const http = require('http');
const express = require('express');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.raw({ type: 'image/jpeg', limit: '10mb' })); // Add raw binary support

API_KEY = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Cropping control flags
let croppingMode = {
  NoCrop: true,
  CropLeft: false,
  CropMiddle: false,
  CropRight: false,
  CropCards: false
};

// ROI definitions
const ROI_DEFINITIONS = {
  CropLeft: { x: 0, y: 0, width: 213, height: 480, section: 'LEFT' },
  CropMiddle: { x: 213, y: 0, width: 214, height: 480, section: 'MIDDLE' },
  CropRight: { x: 427, y: 0, width: 213, height: 480, section: 'RIGHT' },
  CropCards: { x: 20, y: 90, width: 600, height: 300, section: 'CARDS' }
};

// Try to load sharp
let sharp;
try {
  sharp = require('sharp');
  console.log('✅ Sharp loaded - image cropping enabled');
} catch(e) {
  console.log('⚠️  Sharp not installed - install with: npm install sharp');
}

// Create uploads directory
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

// Set cropping mode endpoint
app.post('/api/set-crop-mode', validateApiKey, (req, res) => {
  const { NoCrop, CropLeft, CropMiddle, CropRight, CropCards } = req.body;
  
  croppingMode = {
    NoCrop: false,
    CropLeft: false,
    CropMiddle: false,
    CropRight: false,
    CropCards: false
  };
  
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

// Get current cropping mode
app.get('/api/get-crop-mode', validateApiKey, (req, res) => {
  res.json({ mode: croppingMode });
});

// NEW: Raw binary image upload endpoint
app.post('/api/upload-image-raw', validateApiKey, async (req, res) => {
  try {
    console.log('📸 Raw image upload request received');
    
    // Get metadata from headers
    const device_id = req.headers['x-device-id'] || 'ESP32';
    const timestamp = req.headers['x-timestamp'] || Date.now();
    const image_width = parseInt(req.headers['x-image-width']) || 640;
    const image_height = parseInt(req.headers['x-image-height']) || 480;
    
    // Body is already binary buffer
    const imageBuffer = req.body;
    
    if (!imageBuffer || imageBuffer.length === 0) {
      return res.status(400).json({ message: 'Missing image data' });
    }
    
    console.log(`📦 Received: ${imageBuffer.length} bytes (${image_width}x${image_height})`);
    
    // Validate JPEG
    if (imageBuffer[0] !== 0xFF || imageBuffer[1] !== 0xD8) {
      console.log('❌ Invalid JPEG header');
      return res.status(400).json({ message: 'Invalid JPEG data' });
    }
    
    let finalBuffer = imageBuffer;
    let finalWidth = image_width;
    let finalHeight = image_height;
    let roiSection = '';
    let shouldCrop = false;
    let targetROI = null;
    
    // Determine ROI based on cropping mode
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
    
    // Perform cropping if needed
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
      }
    } else if (shouldCrop && !sharp) {
      console.log('⚠️  Cropping requested but Sharp not installed - saving full image');
      roiSection = `_${targetROI ? targetROI.section : 'UNKNOWN'}_FULL`;
    }
    
    // Generate filename
    const filename = `${device_id}${roiSection}_${timestamp}.jpg`;
    const filepath = path.join(uploadsDir, filename);
    
    // Save to disk
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

// OLD: Base64 JSON endpoint (keep for backwards compatibility)
app.post('/api/upload-image', validateApiKey, async (req, res) => {
  try {
    console.log('📸 Image upload request received (base64 mode)');
    
    const { device_id, timestamp, image_format, image_width, image_height, image_size, image_data, roi } = req.body;
    
    if (!image_data) {
      return res.status(400).json({ message: 'Missing image_data' });
    }

    const imageBuffer = Buffer.from(image_data, 'base64');
    
    let finalBuffer = imageBuffer;
    let finalWidth = image_width;
    let finalHeight = image_height;
    let roiSection = '';
    let shouldCrop = false;
    let targetROI = null;
    
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
      }
    } else if (shouldCrop && !sharp) {
      console.log('⚠️  Cropping requested but Sharp not installed - saving full image');
      roiSection = `_${targetROI ? targetROI.section : 'UNKNOWN'}_FULL`;
    }
    
    const filename = `${device_id || 'ESP32'}${roiSection}_${timestamp || Date.now()}.${image_format || 'jpg'}`;
    const filepath = path.join(uploadsDir, filename);
    
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

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: Date.now(),
    cropMode: croppingMode
  });
});

// Start server
http.createServer(app).listen(3000, '0.0.0.0', () => {
  console.log('HTTP server running on port 3000');
  console.log('Endpoints:');
  console.log('  POST /api/upload-image-raw - Upload camera images (RAW BINARY - RECOMMENDED)');
  console.log('  POST /api/upload-image - Upload camera images (base64 JSON - legacy)');
  console.log('  POST /api/set-crop-mode - Set cropping mode');
  console.log('  GET  /api/get-crop-mode - Get current cropping mode');
  console.log('  GET  /api/health - Health check');
  console.log(`Images will be saved to: ${uploadsDir}`);
  console.log('\n🔧 Default crop mode:', croppingMode);
});