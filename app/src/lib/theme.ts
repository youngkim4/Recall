// Appearance preference. 'system' leaves color-scheme to the OS; the other two
// pin it through data-theme on <html>. index.html applies the stored value
// before first paint, using the same key.

export type ThemePreference = 'system' | 'light' | 'dark'

const THEME_STORAGE_KEY = 'recall.theme'

export const themeOptions: Array<{ value: ThemePreference; label: string }> = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]

export function readThemePreference(): ThemePreference {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY)
    return stored === 'light' || stored === 'dark' ? stored : 'system'
  } catch {
    return 'system'
  }
}

export function applyThemePreference(preference: ThemePreference) {
  const root = document.documentElement
  if (preference === 'system') {
    delete root.dataset.theme
  } else {
    root.dataset.theme = preference
  }
}

export function storeThemePreference(preference: ThemePreference) {
  try {
    if (preference === 'system') {
      window.localStorage.removeItem(THEME_STORAGE_KEY)
    } else {
      window.localStorage.setItem(THEME_STORAGE_KEY, preference)
    }
  } catch {
    // the preference still applies for this session
  }
}
