FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY azure-api/package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY azure-api/ ./

# Create directory for SSL certificates
RUN mkdir -p /app/certs

# Create a non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Change ownership of the app directory
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expose ports
EXPOSE 3000 3443

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js

# Start the application
CMD ["node", "server.js"]