import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Plus, MessageSquare, Trash2 } from 'lucide-react'
import { sessionsApi } from '../api/sessions'
import { useSessionStore } from '../stores/useSessionStore'
import { useUserStore } from '../stores/useUserStore'
import type { ChatSession } from '../types/api'
import EmptyState from '../components/common/EmptyState'
import ConfirmDialog from '../components/common/ConfirmDialog'

export default function Sessions({ embedded = false, onSelect }: { embedded?: boolean; onSelect?: () => void }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const userId = useUserStore((s) => s.userInfo?.uuid || s.userInfo?.user_id || s.userInfo?.id || '')
  const { sessions, setSessions, removeSession, setLoading, loading } = useSessionStore()
  const [deleteTarget, setDeleteTarget] = useState<ChatSession | null>(null)

  const loadSessions = useCallback(async () => {
    if (!userId) return
    setLoading(true)
    try {
      const res = await sessionsApi.list(userId as string)
      const sessionList = (res.data as { sessions: ChatSession[] } | undefined)?.sessions || []
      setSessions(sessionList as ChatSession[])
    } catch {
      toast.error('加载会话列表失败')
    } finally {
      setLoading(false)
    }
  }, [userId, setLoading, setSessions])

  useEffect(() => { loadSessions() }, [loadSessions])

  const handleDelete = async () => {
    if (!deleteTarget) return
    try {
      await sessionsApi.delete(deleteTarget.id)
      removeSession(deleteTarget.id)
    } catch {
      toast.error('删除会话失败')
    }
    setDeleteTarget(null)
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '—'
    const d = new Date(dateStr)
    if (Number.isNaN(d.getTime())) return '—'
    return d.toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })
  }

  return (
    <div className={`max-w-3xl mx-auto px-6 ${embedded ? 'py-4' : 'py-8'}`}>
      <div className="flex items-center justify-between mb-6">
        {!embedded && <h1 className="font-heading text-xl font-semibold text-[var(--color-text)]">{t('nav.sessions')}</h1>}
        <button
          onClick={() => {
            sessionStorage.removeItem('lastSessionId')
            navigate('/chat')
            onSelect?.()
          }}
          className="flex items-center gap-2 px-4 py-2 rounded-md bg-[var(--color-action)] text-white text-sm hover:bg-[var(--color-action-hover)] transition-colors"
        >
          <Plus size={16} />
          {t('chat.newSession')}
        </button>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-[var(--color-bg-tertiary)] rounded-lg animate-pulse" />
          ))}
        </div>
      ) : sessions.length === 0 ? (
        <EmptyState
          icon={<MessageSquare size={48} />}
          message={t('common.noMore')}
          action={
            <button
              onClick={() => {
                sessionStorage.removeItem('lastSessionId')
                navigate('/chat')
                onSelect?.()
              }}
              className="px-4 py-2 text-sm rounded-md bg-[var(--color-action)] text-white"
            >
              {t('chat.newSession')}
            </button>
          }
        />
      ) : (
        <div className="space-y-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              className="flex items-center justify-between gap-2 px-4 py-3 rounded-lg bg-[var(--color-card)] border border-[var(--color-border)] hover:border-[var(--color-accent)] transition-colors"
            >
              <button
                onClick={() => { navigate(`/chat/${session.id}`); onSelect?.() }}
                title={session.title || t('chat.newSession')}
                className="flex flex-1 items-center gap-3 min-w-0 text-left"
              >
                <MessageSquare size={16} className="text-[var(--color-text-tertiary)] shrink-0" />
                <span className="text-sm text-[var(--color-text)] truncate">{session.title || t('chat.newSession')}</span>
              </button>
              <div className="flex items-center gap-3 shrink-0">
                <time
                  dateTime={session.updated_at || session.created_at || undefined}
                  title="最近对话时间（本地时间）"
                  className="text-xs text-[var(--color-text-tertiary)]"
                >
                  {formatDate(session.updated_at || session.created_at)}
                </time>
                <button
                  aria-label={`删除对话 ${session.title || t('chat.newSession')}`}
                  onClick={(e) => { e.stopPropagation(); setDeleteTarget(session) }}
                  className="p-1 rounded text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] hover:bg-[var(--color-danger-bg)] transition-colors"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={() => setDeleteTarget(null)}
        title={t('common.confirm')}
        message="确定要删除这段对话吗？删除后无法恢复。"
        variant="danger"
        confirmText={t('note.delete')}
        onConfirm={handleDelete}
      />
    </div>
  )
}
