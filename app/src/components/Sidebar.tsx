import {
  AnalyzeIcon,
  ChatIcon,
  HomeIcon,
  JobsIcon,
  ReportIcon,
  SearchIcon,
  SettingsIcon,
  SidebarIcon,
} from './Icons'
import type { CSSProperties } from 'react'
import type { ViewKey } from '../types'
import { formatNumber } from '../lib/format'

type SidebarProps = {
  activeView: ViewKey
  activeJobCount: number
  collapsed: boolean
  // false while the window is too narrow for the open rail
  canExpand: boolean
  onToggleCollapse: () => void
  onViewChange: (view: ViewKey) => void
}

type NavEntry = {
  key: ViewKey
  label: string
  icon: typeof HomeIcon
  // a number beside a destination reads as work waiting there, so only
  // running jobs get one
  count?: number
}

export function Sidebar({
  activeView,
  activeJobCount,
  collapsed,
  canExpand,
  onToggleCollapse,
  onViewChange,
}: SidebarProps) {
  const primary: NavEntry[] = [
    { key: 'home', label: 'Overview', icon: HomeIcon },
    { key: 'ask', label: 'Chat', icon: ChatIcon },
    { key: 'analyze', label: 'Analyze', icon: AnalyzeIcon },
    { key: 'explore', label: 'Search', icon: SearchIcon },
    { key: 'reports', label: 'Reports', icon: ReportIcon },
    { key: 'jobs', label: 'Jobs', icon: JobsIcon, count: activeJobCount },
  ]
  const secondary: NavEntry[] = [{ key: 'settings', label: 'Settings', icon: SettingsIcon }]

  function renderItem(item: NavEntry) {
    const Icon = item.icon
    const active = activeView === item.key
    return (
      <button
        key={item.key}
        type="button"
        className="nav-item"
        aria-current={active ? 'page' : undefined}
        // folded, the words are gone, so the name is given here, with the number the row would show
        aria-label={collapsed ? (item.count ? `${item.label}, ${formatNumber(item.count)} running` : item.label) : undefined}
        title={collapsed ? item.label : undefined}
        onClick={() => onViewChange(item.key)}
      >
        <Icon filled={active} />
        <span className="nav-label">{item.label}</span>
        {item.count ? <span className="nav-count">{formatNumber(item.count)}</span> : null}
      </button>
    )
  }

  // Where the fill behind the current row sits in each list. When the row is
  // in the other list, the fill waits one place past the end nearest it.
  const inPrimary = primary.findIndex((item) => item.key === activeView)
  const inSecondary = secondary.findIndex((item) => item.key === activeView)

  return (
    <div className="rail">
      <nav
        className="nav"
        aria-label="Primary"
        data-current={inPrimary >= 0 || undefined}
        style={{ '--at': inPrimary >= 0 ? inPrimary : primary.length } as CSSProperties}
      >
        {primary.map(renderItem)}
      </nav>
      <div className="rail-foot">
        <nav
          className="nav"
          aria-label="App"
          data-current={inSecondary >= 0 || undefined}
          style={{ '--at': inSecondary >= 0 ? inSecondary : -1 } as CSSProperties}
        >
          {secondary.map(renderItem)}
        </nav>
        {canExpand ? (
          <button
            type="button"
            className="icon-btn"
            aria-label={collapsed ? 'Show sidebar' : 'Hide sidebar'}
            aria-expanded={!collapsed}
            title={collapsed ? 'Show sidebar' : 'Hide sidebar'}
            onClick={onToggleCollapse}
          >
            <SidebarIcon />
          </button>
        ) : null}
      </div>
    </div>
  )
}
