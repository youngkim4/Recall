import { useState } from 'react'
import { formatDate, formatLongDate, formatNumber, plural } from '../../lib/format'
import {
  applyThemePreference,
  readThemePreference,
  storeThemePreference,
  themeOptions,
  type ThemePreference,
} from '../../lib/theme'
import type { ArchiveSummary, Defaults, PreviewCacheSummary, RuntimePaths } from '../../types'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { Disclosure } from '../ui/Disclosure'
import { QualityControl } from '../ui/QualityControl'
import { FileLocations } from './FileLocations'
import { MeaningIndex } from './MeaningIndex'

type SettingsViewProps = {
  defaults: Defaults | null
  archive: ArchiveSummary | null
  defaultPaths: RuntimePaths
  paths: RuntimePaths
  model: string
  contactNameCount: number
  cacheStats: PreviewCacheSummary
  // true while the app is reading contact names or changing file locations
  busy: boolean
  // what that work last said
  statusMessage: string
  onModelChange: (model: string) => void
  onApplyPaths: (paths: RuntimePaths) => void
  onRefreshContacts: () => void
  onClearPreviewCache: () => void
  // opens the guide that builds the archive, or updates one that exists
  onOpenSetup: () => void
}

export function SettingsView({
  defaults,
  archive,
  defaultPaths,
  paths,
  model,
  contactNameCount,
  cacheStats,
  busy,
  statusMessage,
  onModelChange,
  onApplyPaths,
  onRefreshContacts,
  onClearPreviewCache,
  onOpenSetup,
}: SettingsViewProps) {
  const [theme, setTheme] = useState<ThemePreference>(() => readThemePreference())
  const names = contactNameCount || defaults?.contactNames?.count || 0
  const namesUpdated = defaults?.contactNames?.exportedAt || defaults?.contactNames?.updatedAt

  function chooseTheme(next: string) {
    const preference = themeOptions.find((option) => option.value === next)?.value ?? 'system'
    setTheme(preference)
    applyThemePreference(preference)
    storeThemePreference(preference)
  }

  return (
    <div className="view">
      <div className="page wide settings-page arrive-each">
        {/* how the app looks and thinks */}
        <div className="settings-col">
          <section className="section ruled" aria-labelledby="settings-look">
            <div className="section-head">
              <h2 id="settings-look">Appearance</h2>
            </div>
            <div className="settings">
              <div className="setting">
                <span className="setting-label">Theme</span>
                <span className="setting-value" />
                <ChoiceMenu label="Theme" value={theme} choices={themeOptions} onChange={chooseTheme} align="end" quietLabel />
              </div>
            </div>
          </section>

          <section className="section ruled" aria-labelledby="settings-ai">
            <div className="section-head">
              <h2 id="settings-ai">AI</h2>
            </div>
            <div className="settings">
              <div className="setting">
                <span className="setting-label">Quality</span>
                <span className="setting-value" />
                <QualityControl models={defaults?.models ?? []} value={model} onChange={onModelChange} align="end" quietLabel />
              </div>
              <MeaningIndex
                messagesPath={paths.messagesPath || defaults?.messagesPath || ''}
                refreshKey={`${archive?.messages} ${archive?.lastMessage}`}
              />
            </div>
          </section>
        </div>

        {/* the archive and where it is kept */}
        <div className="settings-col">
          <section className="section ruled" aria-labelledby="settings-archive">
            <div className="section-head">
              <h2 id="settings-archive">Archive</h2>
            </div>
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
          </section>

          <section className="section ruled">
            <Disclosure className="setting-more" summary="File locations">
              <FileLocations paths={paths} defaultPaths={defaultPaths} busy={busy} onApply={onApplyPaths} />
            </Disclosure>
          </section>
        </div>
      </div>
    </div>
  )
}
