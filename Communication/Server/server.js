const http = require('http');
const express = require('express');

const app = express();
app.use(express.json());

const API_KEY = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Basic signal endpoint
app.post('/api/signal', (req, res) => {
  const clientApiKey = req.headers['x-api-key'];
  if (clientApiKey !== API_KEY) {
    return res.status(403).json({ message: 'Forbidden: Invalid API Key' });
  }

  console.log('Signal received:', req.body);
  res.json({ message: 'Signal received successfully!' });
});

// Start HTTP server
http.createServer(app).listen(3000, '0.0.0.0', () => {
  console.log('HTTP server running on port 3000');
});