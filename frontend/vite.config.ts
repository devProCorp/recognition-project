import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/calibrar': 'http://localhost:5050',
      '/analizar': 'http://localhost:5050',
      '/estado_calibracion': 'http://localhost:5050',
      '/historial': 'http://localhost:5050',
      '/vocabulario': 'http://localhost:5050',
      '/segmentar': 'http://localhost:5050',
      '/segmento': 'http://localhost:5050',
      '/audio_respuesta': 'http://localhost:5050',
    },
  },
  build: {
    outDir: '../static/react-dist',
    emptyOutDir: true,
  },
})
