import { useCallback, useEffect, useRef, useState } from 'react'
import { readStoredLayout, storeLayout, type LayoutState } from './appStorage'

// How wide the rail and the lists are, and whether the rail is folded. Each
// change is remembered for the next launch.
export function useLayout() {
  const [layout, setLayout] = useState<LayoutState>(() => readStoredLayout())
  const remembered = useRef(layout)

  // Remembered once a change has been drawn, not while the next layout is
  // being worked out. The layout the app opened with is not written back: a
  // pane nobody has dragged keeps following its first size.
  useEffect(() => {
    if (remembered.current === layout) return
    remembered.current = layout
    storeLayout(layout)
  }, [layout])

  const change = useCallback((patch: Partial<LayoutState>) => {
    setLayout((current) => ({ ...current, ...patch }))
  }, [])

  const toggleSidebar = useCallback(() => {
    setLayout((current) => ({ ...current, sidebarCollapsed: !current.sidebarCollapsed }))
  }, [])

  return { layout, change, toggleSidebar }
}
