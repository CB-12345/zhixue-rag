import { Outlet } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

export default function AuthLayout() {
  const { t } = useTranslation()

  return (
    <main className="auth-shell min-h-dvh flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-md mx-auto">
        <div className="auth-card rounded-3xl border border-[var(--color-border)] bg-[var(--color-card)] p-7 sm:p-10">
          <Outlet />
        </div>
        <p className="mt-6 text-center text-xs text-[var(--color-text-secondary)]">{t('app.tagline')}</p>
      </div>
    </main>
  )
}
