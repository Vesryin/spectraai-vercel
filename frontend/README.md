# Spectra AI Frontend

Modern React + TypeScript + Tailwind CSS frontend for Spectra AI.

## Quick Start

1. **Install Node.js** (if not installed):

   ```bash
   winget install OpenJS.NodeJS
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Start development server**:

   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## Features

- ⚡ **Vite** for fast development and building
- ⚛️ **React 18** with TypeScript
- 🎨 **Tailwind CSS** for styling
- 🎭 **Framer Motion** for animations
- 💫 **Lucide React** for icons
- 📡 **Axios** for API communication

## Development

The frontend is configured to proxy API requests to the Flask backend running on port 5000.

Make sure the backend server is running before starting the frontend.

## Deployment

This frontend can be deployed to:

- GitHub Pages
- Netlify
- Vercel
- Any static hosting service

The build command generates optimized static files in the `dist/` directory.
