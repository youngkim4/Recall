import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'
import { moveWindowFromItsTop } from './lib/windowDrag.ts'

// In the packaged macOS app the content sits under the native traffic lights;
// flag it so the sidebar can leave room (no wasted space in the browser).
if (new URLSearchParams(window.location.search).get('app') === 'mac') {
  document.documentElement.classList.add('is-mac-app')
  const stopMoving = moveWindowFromItsTop()
  // a module that is replaced while the app runs takes its listener with it
  import.meta.hot?.dispose(stopMoving)
}

// The interface is built on light-dark() colors and the popover attribute.
// A web engine without them would draw it with no colors and every menu open,
// so it is told plainly instead. (Safari 17.5: macOS 14.5, or an updated Safari.)
const engineIsNewEnough =
  typeof CSS !== 'undefined' && CSS.supports('color', 'light-dark(#000, #fff)') && 'popover' in HTMLElement.prototype

const root = document.getElementById('root')!

if (engineIsNewEnough) {
  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>,
  )
} else {
  const note = document.createElement('p')
  note.textContent = 'Recall needs a newer web engine than this Mac has. Update macOS or Safari, then open Recall again.'
  note.style.cssText = 'max-width: 28rem; margin: 20vh auto 0; padding: 0 2rem; font: 15px/1.6 -apple-system, sans-serif;'
  root.append(note)
}
