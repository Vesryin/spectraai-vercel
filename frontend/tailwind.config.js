/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        spectra: {
          primary: '#8B5CF6',
          secondary: '#A78BFA', 
          accent: '#C4B5FD',
          light: '#F3F4F6',
          dark: '#1F2937',
          success: '#10B981',
          warning: '#F59E0B',
          error: '#EF4444',
        },
        gradient: {
          start: '#667eea',
          end: '#764ba2',
        }
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'display': ['Poppins', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-soft': 'pulseSoft 2s infinite',
        'typing': 'typing 1s steps(20) infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        typing: {
          '0%': { opacity: '0.3' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
