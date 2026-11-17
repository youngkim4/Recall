import { useCallback, useSyncExternalStore } from 'react'

// Whether a CSS media query matches, kept current as the window changes.
export function useMediaQuery(query: string) {
  // the same function from one render to the next, so a query is listened to once
  const subscribe = useCallback(
    (notify: () => void) => {
      const list = window.matchMedia(query)
      list.addEventListener('change', notify)
      return () => list.removeEventListener('change', notify)
    },
    [query],
  )
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches)
}
