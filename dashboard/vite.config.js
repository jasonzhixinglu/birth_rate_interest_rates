import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// On GitHub Pages the app is served from https://<user>.github.io/<repo>/,
// so the asset base must be the repo name. Locally it stays at root.
export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_ACTIONS ? '/birth_rate_interest_rates/' : '/',
})
