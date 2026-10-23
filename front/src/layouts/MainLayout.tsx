import { useState } from 'react'
import { Outlet, Navigate } from 'react-router-dom'
import Sidebar from '../components/layout/Sidebar'
import { useUserStore } from '../stores/useUserStore'

export default function MainLayout() {
  const isLogin = useUserStore((s) => s.isLogin)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.matchMedia('(max-width: 767px)').matches)

  if (!isLogin) {
    return <Navigate to="/login" replace />
  }

  return (
    <div className="app-shell flex h-dvh overflow-hidden bg-[var(--color-bg)]">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((v) => !v)}
      />
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
