import { useId, useState, type CSSProperties } from 'react'
import { formatDate, formatLongDate, formatNumber, plural } from '../../lib/format'
import { applyPalette, readPalette, storePalette, type Palette } from '../../lib/palette'
import {
  applyThemePreference,
  readThemePreference,
  storeThemePreference,
  themeOptions,
  type ThemePreference,
} from '../../lib/theme'
import { useZoomSheet } from '../../lib/useZoomSheet'
import type { ArchiveSummary, Defaults, PreviewCacheSummary, RuntimePaths } from '../../types'
import { AppearanceIcon, ArchiveIcon, CloseIcon, FolderIcon } from '../Icons'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { PalettePicker } from '../ui/PalettePicker'
import { ChatSettings } from './ChatSettings'
import { FileLocations } from './FileLocations'

type SettingsSheetProps = {
  defaults: Defaults | null
  archive: ArchiveSummary | null
  defaultPaths: RuntimePaths
  paths: RuntimePaths
  contactNameCount: number
  cacheStats: PreviewCacheSummary
  // true while the app is reading contact names or changing file locations
  busy: boolean
  // what that work last said
  statusMessage: string
  onApplyPaths: (paths: RuntimePaths) => void
  onRefreshContacts: () => void
  onClearPreviewCache: () => void
  // opens the guide that builds the archive, or updates one that exists
  onOpenSetup: () => void
  onClose: () => void
  // the Settings button of the rail: the sheet grows out of it
  from?: Element | null
}

type Part = 'appearance' | 'archive' | 'files'

const PARTS: Array<{ key: Part; label: string; icon: typeof ArchiveIcon }> = [
  { key: 'appearance', label: 'Appearance', icon: AppearanceIcon },
  { key: 'archive', label: 'Archive', icon: ArchiveIcon },
  { key: 'files', label: 'Files', icon: FolderIcon },
]

// Settings as a sheet over the view in use, after ChatGPT's and Notion's: the
// parts in a list on its left, the part chosen on its right, as rows of a
// label, what it is set to and one way to change it. It grows out of the
// Settings button of the rail and goes back into it. The user chose this from
// three mockups; there is no AI part: Quality is set where it is used, in Chat.
export function SettingsSheet(props: SettingsSheetProps) {
  const { close, dialogProps } = useZoomSheet(props.from, props.onClose)
  const [part, setPart] = useState<Part>('appearance')
  const titleId = useId()
  const chosen = PARTS.find((one) => one.key === part) ?? PARTS[0]

  return (
    <dialog {...dialogProps} className="sheet settings-sheet" aria-label="Settings">
      <div className="settings-side">
        <button type="button" className="icon-btn" aria-label="Close" onClick={() => void close()}>
          <CloseIcon />
        </button>
        {/* the rail's list: its fill slides to the part chosen */}
        <nav
          className="nav"
          aria-label="Settings"
          data-current
          style={{ '--at': PARTS.indexOf(chosen) } as CSSProperties}
        >
          {PARTS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              type="button"
              className="nav-item"
              aria-current={key === part ? 'page' : undefined}
              onClick={() => setPart(key)}
            >
              <Icon />
              <span className="nav-label">{label}</span>
            </button>
          ))}
        </nav>
      </div>
      <div className="settings-pane">
        {/* a part that is chosen comes in as a view does */}
        <section key={part} className="settings-part arrive" aria-labelledby={titleId}>
          <h2 id={titleId} className="settings-title">
            {chosen.label}
          </h2>
          {part === 'appearance' ? <AppearancePart /> : part === 'archive' ? <ArchivePart {...props} /> : null}
          {part === 'files' ? (
            <FileLocations
              // applied locations start a new draft
              key={`${props.paths.messagesPath}\n${props.paths.outDir}`}
              paths={props.paths}
              defaultPaths={props.defaultPaths}
              busy={props.busy}
              onApply={props.onApplyPaths}
            />
          ) : null}
        </section>
      </div>
    </dialog>
  )
}

// light or dark, and the colours of the app
function AppearancePart() {
  const [theme, setTheme] = useState<ThemePreference>(() => readThemePreference())
  const [palette, setPalette] = useState<Palette>(() => readPalette())
  const colourId = useId()

  function chooseTheme(next: string) {
    const preference = themeOptions.find((option) => option.value === next)?.value ?? 'system'
    setTheme(preference)
    applyThemePreference(preference)
    storeThemePreference(preference)
  }

  function choosePalette(next: Palette) {
    setPalette(next)
    applyPalette(next)
    storePalette(next)
  }

  return (
    <div className="settings">
      <div className="setting">
        <span className="setting-label">Theme</span>
        <span className="setting-value" />
        <ChoiceMenu label="Theme" value={theme} choices={themeOptions} onChange={chooseTheme} align="end" quietLabel />
      </div>
      <div className="setting">
        <span className="setting-label" id={colourId}>
          Colour
        </span>
        {/* no name here: the chosen tile is ringed and named under itself */}
        <span className="setting-value" />
        <PalettePicker value={palette} onChange={choosePalette} labelledBy={colourId} />
      </div>
    </div>
  )
}

// what the archive holds, and the ways to bring it up to date
function ArchivePart({
  defaults,
  archive,
  contactNameCount,
  cacheStats,
  busy,
  statusMessage,
  onRefreshContacts,
  onClearPreviewCache,
  onOpenSetup,
}: SettingsSheetProps) {
  const names = contactNameCount || defaults?.contactNames?.count || 0
  const namesUpdated = defaults?.contactNames?.exportedAt || defaults?.contactNames?.updatedAt
  return (
    <>
      {statusMessage ? (
        <p className="notice" role="status">
          {statusMessage}
        </p>
      ) : null}
      <div className="settings">
        <div className="setting">
          <span className="setting-label">Messages</span>
          <span className="setting-value">
            {archive?.conversations
              ? `${formatNumber(archive.messages)}, imported through ${formatLongDate(archive.lastMessage)}`
              : 'None imported'}
          </span>
          <button type="button" className="btn" disabled={busy} onClick={onOpenSetup}>
            {archive?.conversations ? 'Update' : 'Build'}
          </button>
        </div>
        <ChatSettings messagesPath={defaults?.messagesPath ?? ''} />
        <div className="setting">
          <span className="setting-label">Contact names</span>
          <span className="setting-value">
            {names ? `${formatNumber(names)}${namesUpdated ? `, read ${formatDate(namesUpdated)}` : ''}` : 'None read'}
          </span>
          <button type="button" className="btn" disabled={busy} onClick={onRefreshContacts}>
            Read again
          </button>
        </div>
        <div className="setting">
          <span className="setting-label">Saved figures</span>
          <span className="setting-value">{plural(cacheStats.count, 'conversation')}</span>
          <button type="button" className="btn" disabled={!cacheStats.count} onClick={onClearPreviewCache}>
            Clear
          </button>
        </div>
      </div>
    </>
  )
}
