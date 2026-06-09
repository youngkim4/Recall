import {
  ClockIcon,
  CompassIcon,
  FileIcon,
  HomeIcon,
  LayersIcon,
  MessageIcon,
  PanelLeftIcon,
  SettingsIcon,
} from './Icons'
import type { Defaults, ViewKey } from '../types'
import { formatNumber } from '../lib/format'

type SidebarProps = {
  activeView: ViewKey
  defaults: Defaults | null
  namedCount: number
  conversationCount: number
  reportCount: number
  collapsed: boolean
  onToggleCollapse: () => void
  onViewChange: (view: ViewKey) => void
}

const navItems: Array<{ key: ViewKey; label: string; icon: typeof LayersIcon }> = [
  { key: 'home', label: 'Overview', icon: HomeIcon },
  { key: 'ask', label: 'Chat', icon: MessageIcon },
  { key: 'analyze', label: 'Analyze', icon: LayersIcon },
  { key: 'explore', label: 'Search', icon: CompassIcon },
  { key: 'reports', label: 'Reports', icon: FileIcon },
  { key: 'jobs', label: 'Jobs', icon: ClockIcon },
  { key: 'settings', label: 'Settings', icon: SettingsIcon },
]

export function Sidebar({
  activeView,
  defaults,
  namedCount,
  conversationCount,
  reportCount,
  collapsed,
  onToggleCollapse,
  onViewChange,
}: SidebarProps) {
  return (
    <aside className={`sidebar ${collapsed ? 'is-collapsed' : ''}`}>
      <div className="sidebar-head">
        <button
          type="button"
          className="sidebar-toggle"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!collapsed}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          onClick={onToggleCollapse}
        >
          <PanelLeftIcon className="nav-icon" />
        </button>
      </div>

      <nav className="side-nav" aria-label="Primary">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.key}
              type="button"
              className={`side-nav-item ${activeView === item.key ? 'active' : ''}`}
              title={collapsed ? item.label : undefined}
              onClick={() => onViewChange(item.key)}
            >
              <Icon className="nav-icon" />
              <span>{item.label}</span>
            </button>
          )
        })}
      </nav>

      <div className="sidebar-section">
        <span className="section-label">Library</span>
        <dl className="side-metrics">
          <div>
            <dt>Conversations</dt>
            <dd>{formatNumber(conversationCount)}</dd>
          </div>
          <div>
            <dt>Named</dt>
            <dd>{formatNumber(namedCount || defaults?.contactNames?.count || 0)}</dd>
          </div>
          <div>
            <dt>Reports</dt>
            <dd>{formatNumber(reportCount)}</dd>
          </div>
        </dl>
      </div>
    </aside>
  )
}
