import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'

// In the packaged macOS app the content sits under the native traffic lights;
// flag it so the sidebar can leave room (no wasted space in the browser).
if (new URLSearchParams(window.location.search).get('app') === 'mac') {
  document.documentElement.classList.add('is-mac-app')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
