import { useState } from 'react'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  FileText,
  MessageSquare,
  Ellipsis,
  ChevronDown,
  GraduationCap,
  Library,
  Network,
  Settings,
  User,
  Info,
  LogOut,
  Columns2,
} from 'lucide-react'
import { useUserStore } from '../../stores/useUserStore'
import { authApi } from '../../api/auth'
import BrandMark from '../common/BrandMark'

const navItems = [
  { path: '/knowledge', icon: Library, labelKey: 'nav.knowledge' },
  { path: '/chat', icon: MessageSquare, labelKey: 'nav.chat' },
  { path: '/notes', icon: FileText, labelKey: 'nav.notes' },
]

const extraItems = [
  { path: '/review', icon: GraduationCap, labelKey: 'nav.review' },
  { path: '/graph', icon: Network, labelKey: 'nav.graph' },
]

const bottomItems = [
  { path: '/profile', icon: User, labelKey: 'nav.profile' },
  { path: '/settings', icon: Settings, labelKey: 'nav.settings' },
  { path: '/about', icon: Info, labelKey: 'nav.about' },
]

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const [moreOpen, setMoreOpen] = useState(() => extraItems.some((item) => item.path === pathname))
  const showMore = moreOpen
  const logout = useUserStore((s) => s.logout)

  const handleLogout = async () => {
    try { await authApi.logout() } catch { /* ignore */ }
    logout()
    navigate('/login')
  }

  return (
    <aside
      className={`flex flex-col border-r border-[var(--color-border)] bg-[var(--color-sidebar)] shrink-0 transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-60'
      }`}
    >
      <div className={`flex items-center h-24 ${collapsed ? 'justify-center px-2' : 'justify-between px-4'}`}>
        {!collapsed && (
          <div className="flex items-center gap-2.5 min-w-0">
            <BrandMark />
            <div className="min-w-0">
              <h1 className="font-heading text-xl font-semibold tracking-wide text-[var(--color-text)]">{t('app.name')}</h1>
              <p className="mt-0.5 text-[11px] text-[var(--color-text-secondary)] truncate">{t('app.subtitle')}</p>
            </div>
          </div>
        )}
        <button
          onClick={onToggle}
          className="p-1.5 rounded-md text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-secondary)] hover:text-[var(--color-text)] transition-colors"
          title={collapsed ? t('nav.expand') : t('nav.collapse')}
          aria-label={collapsed ? t('nav.expand') : t('nav.collapse')}
        >
          <Columns2 size={18} className={`transition-transform duration-300 ${collapsed ? 'rotate-180' : ''}`} />
        </button>
      </div>

      <nav className="flex-1 min-h-0 overflow-y-auto px-3 space-y-1.5" aria-label={t('app.workspace')}>
        {!collapsed && <p className="px-3 pb-2 text-[11px] font-medium tracking-wider text-[var(--color-text-tertiary)]">{t('app.workspace')}</p>}
        {navItems.map(({ path, icon: Icon, labelKey }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/notes'}
            className={({ isActive }) =>
              `flex items-center gap-3 py-2.5 rounded-lg border text-sm transition-colors ${
                isActive
                  ? 'border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-accent)] font-semibold shadow-sm'
                  : 'border-transparent text-[var(--color-text-secondary)] hover:bg-[var(--color-accent-bg)] hover:text-[var(--color-text)]'
              } ${collapsed ? 'justify-center px-2' : 'px-3'}`
            }
            title={collapsed ? t(labelKey) : undefined}
          >
            <Icon size={18} className="shrink-0" />
            {!collapsed && t(labelKey)}
          </NavLink>
        ))}
        <button
          onClick={() => setMoreOpen((open) => !open)}
          aria-expanded={showMore}
          aria-controls="extra-navigation"
          title={t('nav.more')}
          className={`flex items-center gap-3 py-2.5 rounded-lg text-sm w-full text-[var(--color-text-secondary)] hover:bg-[var(--color-accent-bg)] ${collapsed ? 'justify-center px-2' : 'px-3'}`}
        >
          <Ellipsis size={18} className="shrink-0" />
          {!collapsed && <><span className="flex-1 text-left">{t('nav.more')}</span><ChevronDown size={14} className={showMore ? 'rotate-180' : ''} /></>}
        </button>
        {showMore && <div id="extra-navigation" className="space-y-1">
          {extraItems.map(({ path, icon: Icon, labelKey }) => <NavLink
            key={path} to={path} title={t(labelKey)}
            className={({ isActive }) => `flex items-center gap-3 py-2.5 rounded-lg text-sm ${collapsed ? 'justify-center px-2' : 'pl-6 pr-3'} ${isActive ? 'bg-[var(--color-card)] text-[var(--color-accent)] font-semibold' : 'text-[var(--color-text-tertiary)] hover:bg-[var(--color-accent-bg)]'}`}
          ><Icon size={16} className="shrink-0" />{!collapsed && t(labelKey)}</NavLink>)}
        </div>}
      </nav>

      <div className="px-3 py-3 border-t border-[var(--color-border)] space-y-1">
        {bottomItems.map(({ path, icon: Icon, labelKey }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive }) =>
              `flex items-center gap-3 py-2.5 rounded-lg border text-sm transition-colors ${
                isActive
                  ? 'border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-accent)] font-semibold shadow-sm'
                  : 'border-transparent text-[var(--color-text-secondary)] hover:bg-[var(--color-accent-bg)] hover:text-[var(--color-text)]'
              } ${collapsed ? 'justify-center px-2' : 'px-3'}`
            }
            title={collapsed ? t(labelKey) : undefined}
          >
            <Icon size={18} className="shrink-0" />
            {!collapsed && t(labelKey)}
          </NavLink>
        ))}
        <button
          onClick={handleLogout}
          className={`flex items-center gap-3 py-2.5 rounded-md text-sm w-full text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-secondary)] hover:text-[var(--color-danger)] transition-colors ${
            collapsed ? 'justify-center px-2' : 'px-3'
          }`}
          title={collapsed ? t('nav.logout') : undefined}
        >
          <LogOut size={18} />
          {!collapsed && t('nav.logout')}
        </button>
      </div>
    </aside>
  )
}
