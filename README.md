# Portable Real-Time Poker Assistant

**ECE 1896 Senior Design Project**  
University of Pittsburgh, Fall 2025

## Overview

An ML-powered poker coaching system that provides real-time strategic recommendations during live gameplay. The device combines computer vision, machine learning, and custom hardware to analyze poker hands and suggest optimal actions.

## Team Members

- **PJ Granieri** - Computer Vision & Software Integration Lead
- **Nick Lavine** - Machine Learning Architecture Lead
- **Seth Williams** - Hardware MCU Architecture Lead
- **Ben Wu** - Power Systems & Enclosure Design Lead

## System Architecture

### Hardware
- **MCU**: ESP32-S3-WROOM-1 (dual-core, 240 MHz, 8MB PSRAM)
- **Camera**: OV5640 autofocus (640×480 VGA JPEG capture)
- **Display**: 16×2 LCD with I2C backpack
- **Power**: Custom USB-C PD system with 1S Li-ion battery
  - USB-C PD controller (TPS25730D)
  - Battery charger (BQ25616)
  - 5V boost converter (TPS61023) for LCD
  - 3.3V buck-boost (TPS631010) for MCU/camera

### Software Stack
- **Computer Vision**: 4 specialized YOLOv8-nano models (99%+ mAP)
  - Card detection (52 classes, 99.5% mAP)
  - Pot chips detection (99.5% mAP)
  - Player action detection (99.2% mAP)
  - Player stack analysis
- **Machine Learning**: Custom MLP with 3M+ training scenarios
  - Professional gameplay data
  - Game Theory Optimal (GTO) datasets
  - Bot-simulated hands
- **Cloud Infrastructure**: Azure-hosted FastAPI inference service
- **Firmware**: C++/Arduino on ESP32-S3

## Key Features

✅ Real-time card, chip, and action detection  
✅ Adaptive ROI cropping for bandwidth optimization  
✅ Multi-threshold confidence fusion   
✅ Battery-powered portable operation (3+ hours)  
✅ 3-second end-to-end latency (capture → decision)

## Performance Metrics

### Computer Vision
- **Card Detection**: 99.5% mAP@0.5, perfect precision/recall
- **Chip Detection**: 99.5% mAP@0.5
- **Action Detection**: 99.2% mAP@0.5
- **Processing**: 120-180ms inference (CPU)

### Machine Learning
- **Win Rate**: 25-50% in 4-player simulations
- **Training Accuracy**: ~78%
- **Training Loss**: <0.05

### Hardware
- **Power Efficiency**: 92% (boost), 88% (buck-boost)
- **Battery Life**: 3+ hours continuous operation
- **USB-C Charging**: 5V/2A minimum

## Repository Structure
```
├── src/Firmware/              # ESP32-S3 firmware (C++/Arduino)
├── src/ComputerVision/        # YOLO models & detection algorithms
├── src/MachineLearning/       # MLP poker model & training pipeline
├── src/Communication/         # Azure cloud services & API
├── src/Orchestrator/          # Poker FSM & metadata storage
├── Hardware/                  # PCB schematics & layouts (Altium)
├── src/tests/                 # pytest suites & validation datasets
└── docs/                      # Final report & technical documentation
```
## Quick Start

### Hardware Assembly
1. Power board → MCU board → Camera/LCD
2. Connect 1S Li-ion battery
3. USB-C for charging/programming

### Software Setup
1. Flash ESP32-S3 firmware via USB-C
2. Configure WiFi credentials
3. Deploy Azure inference service
4. Start Node.js communication server

### Usage
1. Power on device
2. Position camera overhead poker table
3. Device auto-captures every 3 seconds
4. LCD displays ML recommendations (Fold/Check/Call/Raise)

## Technical Highlights

- **Multi-Model Fusion**: Combines YOLO + MediaPipe for robust action detection
- **Height-Based Chip Counting**: Aspect ratio analysis for stack estimation
- **Adaptive Cropping**: Dynamic ROI switching reduces bandwidth 50-70%
- **Heuristic Enhancement**: Runtime layer adds strategic unpredictability
- **Polling Architecture**: Robust HTTP communication with auto-reconnect

## Future Work

For commercialization:
- Redesign enclosure for consumer usability
- Add power button and battery status indicators
- Reduce overall size for improved portability

## Documentation

Full technical details available in `ECE_1896_Final_Report.pdf`:
- Complete system architecture
- Hardware design & testing
- CV/ML methodology & results
- Integration strategies

## Live Demo Link:
https://youtu.be/1nT3llKM1yY

---
