import { useCallback, useSyncExternalStore } from 'react'
import { useMediaQuery } from './useMediaQuery'

// An attribute of <html>, kept current. The Appearance and the palette are
// put there by Settings, and by index.html before the first paint.
export function usePageAttribute(name: string): string | null {
  const subscribe = useCallback(
    (notify: () => void) => {
      const watcher = new MutationObserver(notify)
      watcher.observe(document.documentElement, { attributes: true, attributeFilter: [name] })
      return () => watcher.disconnect()
    },
    [name],
  )
  return useSyncExternalStore(subscribe, () => document.documentElement.getAttribute(name))
}

// whether the page is dark now: as Appearance pins it, or as the system has it
export function usePageIsDark(): boolean {
  const pinned = usePageAttribute('data-theme')
  const system = useMediaQuery('(prefers-color-scheme: dark)')
  return pinned === 'dark' || (pinned !== 'light' && system)
}
