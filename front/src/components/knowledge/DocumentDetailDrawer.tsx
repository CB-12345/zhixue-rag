import { useEffect, useRef, useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useTranslation } from 'react-i18next'
import { X, Loader2, AlertCircle, FileText } from 'lucide-react'
import { knowledgeApi } from '../../api/knowledge'
import type { KnowledgeDocumentDetail } from '../../types/api'
import AuthImage from '../common/AuthImage'
import { locateSource } from '../../utils/sourceLocation'
import type { SourceReference } from '../../utils/sourceLocation'

interface DocumentDetailDrawerProps {
  filename: string | null
  onClose: () => void
  reference?: SourceReference | null
}

type Tab = 'content' | 'chunks'

export default function DocumentDetailDrawer({ filename, onClose, reference }: DocumentDetailDrawerProps) {
  const { t } = useTranslation()
  const [tab, setTab] = useState<Tab>('content')
  const [detail, setDetail] = useState<KnowledgeDocumentDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)
  const targetRef = useRef<HTMLDivElement>(null)

  // filename 变化时在 render 阶段同步重置（与原 effect 语义一致，避免 effect 内同步 setState）
  const [prevFilename, setPrevFilename] = useState<string | null | undefined>(undefined)
  if (filename !== prevFilename) {
    setPrevFilename(filename)
    setDetail(null)
    setLoading(!!filename)
    setError(false)
    setTab('content')
  }
  const [prevReference, setPrevReference] = useState<SourceReference | null | undefined>(undefined)
  if (reference !== prevReference) {
    setPrevReference(reference)
    setTab(reference ? 'chunks' : 'content')
  }

  const location = detail ? locateSource(detail, reference) : null
  const targetId = location?.status === 'matched' ? location.chunk.chunk_id : null

  useEffect(() => {
    if (tab === 'chunks' && targetId) {
      targetRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
      targetRef.current?.focus({ preventScroll: true })
    }
  }, [tab, targetId, detail])

  useEffect(() => {
    if (!filename) {
      return
    }
    let cancelled = false
    knowledgeApi.detail(filename)
      .then((res) => { if (!cancelled) setDetail(res.data) })
      .catch(() => { if (!cancelled) setError(true) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [filename])

  return (
    <Dialog.Root open={!!filename} onOpenChange={(open) => { if (!open) onClose() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40" />
        <Dialog.Content className="fixed top-0 right-0 h-full w-[640px] max-w-[90vw] bg-[var(--color-card)] shadow-xl flex flex-col">
          <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--color-border)] shrink-0">
            <Dialog.Title className="text-base font-medium text-[var(--color-text)] truncate flex items-center gap-2">
              <FileText size={16} className="text-[var(--color-text-tertiary)] shrink-0" />
              {detail?.filename || filename || ''}
            </Dialog.Title>
            <Dialog.Description className="sr-only">查看文档全文和切片内容</Dialog.Description>
            <Dialog.Close aria-label="关闭文档" className="p-1.5 rounded text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-secondary)] transition-colors">
              <X size={18} />
            </Dialog.Close>
          </div>

          <div className="flex border-b border-[var(--color-border)] px-6 shrink-0">
            <button
              onClick={() => setTab('content')}
              className="px-4 py-3 text-sm border-b-2 transition-colors"
              style={{
                borderColor: tab === 'content' ? 'var(--color-accent)' : 'transparent',
                color: tab === 'content' ? 'var(--color-accent)' : 'var(--color-text-secondary)',
              }}
            >
              {t('knowledge.detail')}
            </button>
            <button
              onClick={() => setTab('chunks')}
              className="px-4 py-3 text-sm border-b-2 transition-colors"
              style={{
                borderColor: tab === 'chunks' ? 'var(--color-accent)' : 'transparent',
                color: tab === 'chunks' ? 'var(--color-accent)' : 'var(--color-text-secondary)',
              }}
            >
              {t('knowledge.chunks')} ({detail?.chunks?.length || 0})
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {loading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 size={24} className="animate-spin text-[var(--color-text-tertiary)]" />
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center py-20 text-[var(--color-text-tertiary)] gap-2">
                <AlertCircle size={20} />
                <span className="text-sm">{t('common.error')}</span>
                {reference && <p className="text-sm">暂时无法打开当前文档，可返回查看回答中保存的引用片段。</p>}
              </div>
            ) : !detail ? null : tab === 'content' ? (
              <div className="space-y-6">
                {detail.content && (
                  <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap font-sans leading-relaxed">
                    {detail.content}
                  </pre>
                )}
                {detail.images.length > 0 && (
                  <div className="space-y-4">
                    {detail.images.map((img, i) => (
                      <AuthImage
                        key={i}
                        src={img}
                        alt={`Document image ${i + 1}`}
                        className="w-full rounded-lg border border-[var(--color-border)]"
                      />
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                {location?.status === 'matched' && (
                  <p role="status" className="rounded-lg border border-sky-200 bg-sky-50 p-3 text-sm text-sky-800">
                    已定位引用的原文片段{location.chunk.page != null && location.chunk.page > 0 ? ` · 第 ${location.chunk.page} 页` : ''}
                  </p>
                )}
                {location && ['changed', 'missing'].includes(location.status) && (
                  <p role="status" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                    当前文档与这条回答的来源快照不一致，无法定位原引用。可查看当前全文，或返回核对回答中保存的片段。
                  </p>
                )}
                {detail.chunks.map((chunk) => (
                  <div
                    key={chunk.chunk_id}
                    ref={chunk.chunk_id === targetId ? targetRef : undefined}
                    tabIndex={chunk.chunk_id === targetId ? -1 : undefined}
                    aria-current={chunk.chunk_id === targetId ? 'location' : undefined}
                    className={`p-4 rounded-lg border ${chunk.chunk_id === targetId ? 'bg-sky-50 border-sky-400 ring-2 ring-sky-200' : 'bg-[var(--color-bg-secondary)] border-[var(--color-border)]'}`}
                  >
                    <div className="flex items-center gap-2 mb-2 text-xs text-[var(--color-text-tertiary)]">
                      <span className="font-medium text-[var(--color-text-secondary)]">#{chunk.index + 1}</span>
                      {chunk.page != null && chunk.page > 0 && <span>| {t('knowledge.page')} {chunk.page}</span>}
                    </div>
                    <pre className="text-sm text-[var(--color-text)] whitespace-pre-wrap font-sans leading-relaxed">
                      {chunk.content}
                    </pre>
                    {chunk.images.length > 0 && (
                      <div className="mt-3 space-y-2">
                        {chunk.images.map((img, i) => (
                          <AuthImage
                            key={i}
                            src={img}
                            alt={`Chunk ${chunk.index + 1} image ${i + 1}`}
                            className="w-full rounded-lg border border-[var(--color-border)]"
                          />
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
