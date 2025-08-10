# -----------------------
# Stage 1: Build frontend
# -----------------------
FROM node:20-alpine AS builder
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install --production=false

# Copy source code
COPY . .

# Accept API URL at build time (default is prod backend)
ARG VITE_API_URL=https://spectra-ai-library-version-production.up.railway.app
ENV VITE_API_URL=$VITE_API_URL

# Build optimized production files
RUN npm run build

# -----------------------
# Stage 2: Production server
# -----------------------
FROM node:20-alpine AS production
WORKDIR /app

# Install "serve" to host static files
RUN npm install -g serve

# Copy build output from builder stage
COPY --from=builder /app/dist ./dist

# Expose port
EXPOSE 3000

# Default command
CMD ["serve", "-s", "dist", "-l", "3000"]

# -----------------------
# Stage 3: Development (optional)
# -----------------------
# To run locally in dev mode without building, override CMD when starting container:
# docker run -it -p 3000:3000 \
#   -e VITE_API_URL=http://localhost:8000 \
#   spectra-frontend \
#   npm run dev -- --host 0.0.0.0