// Native bridge to the Mac shell -- the only file that touches window.webkit.
// Outside the shell (npm run dev in a browser) every call resolves to null and
// `available` is false, so callers render copy-paste fallbacks instead.

type BridgeResolver = (value: unknown) => void

declare global {
  interface Window {
    __RECALL_SHELL__?: boolean
    __recallBridge?: { _resolve: (id: string, value: unknown) => void }
    webkit?: {
      messageHandlers?: {
        recall?: { postMessage: (body: unknown) => void }
      }
    }
  }
}

const pending = new Map<string, BridgeResolver>()
let counter = 0

function ensureReceiver() {
  if (window.__recallBridge) return
  window.__recallBridge = {
    _resolve(id: string, value: unknown) {
      const resolve = pending.get(id)
      if (resolve) {
        pending.delete(id)
        resolve(value)
      }
    },
  }
}

function post(cmd: string): Promise<unknown> {
  const handler = window.webkit?.messageHandlers?.recall
  if (!handler) return Promise.resolve(null)
  ensureReceiver()
  counter += 1
  const id = `b${counter.toString(36)}`
  return new Promise((resolve) => {
    pending.set(id, resolve)
    handler.postMessage({ cmd, id })
    // a panel dismissed without choosing must not leak the promise forever
    window.setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id)
        resolve(null)
      }
    }, 120000)
  })
}

export const recallBridge = {
  get available(): boolean {
    return Boolean(window.__RECALL_SHELL__ && window.webkit?.messageHandlers?.recall)
  },
  openFullDiskAccess(): void {
    void post('openFullDiskAccess')
  },
  pickDatabaseFile(): Promise<string | null> {
    return post('pickDatabase').then((value) => (typeof value === 'string' && value ? value : null))
  },
  relaunch(): void {
    void post('relaunch')
  },
}
