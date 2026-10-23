import { useEffect, useState, useRef, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Upload, FileText, Trash2, Loader2, CheckCircle2, AlertCircle } from 'lucide-react'
import { knowledgeApi } from '../api/knowledge'
import { useSSE } from '../hooks/useSSE'
import type { KnowledgeSSEMessage, KnowledgeDocument } from '../types/api'
import EmptyState from '../components/common/EmptyState'
import ConfirmDialog from '../components/common/ConfirmDialog'
import DocumentDetailDrawer from '../components/knowledge/DocumentDetailDrawer'

interface UploadFile {
  file: File
  progress: number
  status: 'pending' | 'uploading' | 'success' | 'fail'
  error?: string
}

export default function KnowledgeBase() {
  const { t } = useTranslation()
  const { start: startSSE, loading: uploading } = useSSE()
  const [docs, setDocs] = useState<KnowledgeDocument[]>([])
  const [retrying, setRetrying] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([])
  const [uploadTotal, setUploadTotal] = useState(0)
  const [uploadDone, setUploadDone] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const [showClean, setShowClean] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; filename: string } | null>(null)
  const [detailFilename, setDetailFilename] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadDocs = useCallback(async (background = false) => {
    if (!background) setLoading(true)
    try {
      const res = await knowledgeApi.list()
      const documents = res.data?.documents || []
      setDocs(documents)
    } catch {
      if (!background) toast.error('加载文档列表失败')
    } finally {
      setLoading(false)
    }
  }, [])

  // 挂载时拉取文档列表：loadDocs 首行的 setLoading(true) 仅服务于手动刷新路径（loading 初始值已为 true）；单次提交，无级联渲染
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { loadDocs() }, [loadDocs])

  const processing = docs.some((doc) => ['pending', 'running'].includes(doc.status)
    || ['pending', 'running'].includes(doc.graph_status || ''))
  useEffect(() => {
    if (!processing) return
    const timer = window.setInterval(() => { loadDocs(true) }, 10000)
    return () => window.clearInterval(timer)
  }, [processing, loadDocs])

  const handleFilesSelected = (files: FileList) => {
    if (uploading) return
    const selected = Array.from(files)
    if (selected.some((file) => !/\.(pdf|txt|md|docx|pptx)$/i.test(file.name) || file.size === 0 || file.size > 20 * 1024 * 1024)) {
      toast.error('请选择非空的 PDF、TXT、MD、DOCX 或 PPTX，单个文件不超过 20MB')
      if (fileInputRef.current) fileInputRef.current.value = ''
      return
    }
    if (selected.reduce((sum, file) => sum + file.size, 0) > 200 * 1024 * 1024) {
      toast.error('单次上传总大小不能超过 200MB')
      return
    }
    const newFiles: UploadFile[] = Array.from(files).map((f) => ({ file: f, progress: 0, status: 'pending' }))
    setUploadFiles(newFiles)
    setUploadTotal(newFiles.length)
    setUploadDone(0)

    const formData = new FormData()
    newFiles.forEach((f) => formData.append('files', f.file))

    startSSE(
      '/knowledge/add/multiple/stream',
      formData,
      {
        onKnowledgeProgress: (data: KnowledgeSSEMessage) => {
          const matchesFile = (uf: UploadFile, index: number) => data.file_index
            ? index + 1 === data.file_index : !data.filename || uf.file.name === data.filename
          if (['processing', 'slicing_completed', 'writing'].includes(data.event_type)) {
            setUploadFiles((prev) =>
              prev.map((uf, index) =>
                matchesFile(uf, index)
                  ? { ...uf, progress: data.progress || 0, status: 'uploading' }
                  : uf
              )
            )
          } else if (data.event_type === 'completed') {
            if (data.step === 'existing') toast.info(data.message || '文档已存在，复用已有处理状态')
            setUploadFiles((prev) =>
              prev.map((uf, index) =>
                matchesFile(uf, index)
                  ? { ...uf, progress: 100, status: 'success' }
                  : uf
              )
            )
            setUploadDone((c) => c + 1)
          } else if (data.event_type === 'error') {
            setUploadFiles((prev) =>
              prev.map((uf, index) =>
                matchesFile(uf, index)
                  ? { ...uf, status: 'fail', error: data.error_message || '上传失败' }
                  : uf
              )
            )
            setUploadDone((c) => c + 1)
          } else if (data.event_type === 'finish') {
            loadDocs()
            setUploadFiles((prev) => {
              const allSuccess = prev.length > 0 && prev.every((uf) => uf.status === 'success')
              return allSuccess ? [] : prev
            })
            setUploadTotal(0)
            setUploadDone(0)
          }
        },
        onError: (error) => {
          setUploadFiles((prev) =>
            prev.map((uf) =>
              ['pending', 'uploading'].includes(uf.status) ? { ...uf, status: 'fail' as const, error } : uf
            )
          )
        },
      }
    ).finally(() => {
      if (fileInputRef.current) fileInputRef.current.value = ''
      loadDocs(true)
    })
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      handleFilesSelected(e.dataTransfer.files)
    }
  }

  const handleDelete = async () => {
    if (!deleteTarget) return
    try {
      await knowledgeApi.deleteByFilename(deleteTarget.filename)
      setDocs((prev) => prev.filter((d) => d.id !== deleteTarget.id))
    } catch {
      toast.error('删除文档失败')
    }
    setDeleteTarget(null)
  }

  const handleCleanAll = async () => {
    try {
      await knowledgeApi.cleanAll()
      setDocs([])
    } catch {
      toast.error('清空知识库失败')
    }
    setShowClean(false)
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '时间未知'
    return new Date(dateStr).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="max-w-4xl mx-auto py-8 px-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-heading text-xl font-semibold text-[var(--color-text)]">{t('knowledge.title')}</h1>
        {docs.length > 0 && (
          <button
            onClick={() => setShowClean(true)}
            className="flex items-center gap-2 px-4 py-2 text-sm rounded-md border border-[var(--color-border)] text-[var(--color-danger)] hover:bg-[var(--color-danger-bg)] transition-colors"
          >
            <Trash2 size={14} />
            {t('knowledge.cleanAll')}
          </button>
        )}
      </div>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`learning-card relative border-2 border-dashed rounded-2xl bg-[var(--color-card)] p-10 text-center transition-colors ${
          dragOver ? 'border-[var(--color-accent)] bg-[var(--color-accent-bg)]' : 'border-[var(--color-border)] hover:border-[var(--color-text-tertiary)]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md,.docx,.pptx"
          className="hidden"
          onChange={(e) => e.target.files && handleFilesSelected(e.target.files)}
        />
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--color-accent-bg)] text-[var(--color-accent)]">
          <Upload size={24} />
        </div>
        <p className="text-sm text-[var(--color-text-secondary)] mb-1">{t('knowledge.dragDrop')}</p>
        <p className="text-xs text-[var(--color-text-tertiary)] mb-4">{t('knowledge.fileTypes')}</p>
        <p className="text-xs text-[var(--color-text-tertiary)] mb-4">重复文件会复用已有文档；更新同名文档请先删除旧文件。</p>
        <button
          disabled={uploading}
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 text-sm rounded-md bg-[var(--color-action)] text-white hover:bg-[var(--color-action-hover)] transition-colors"
        >
          {uploading ? '正在接收文件…' : t('knowledge.upload')}
        </button>
      </div>

      {uploadFiles.length > 0 && (
        <div className="mt-4 space-y-2">
          {uploadFiles.map((uf, i) => (
            <div key={i} className="flex items-center gap-3 px-4 py-3 rounded-lg bg-[var(--color-card)] border border-[var(--color-border)]">
              {uf.status === 'success' ? (
                <CheckCircle2 size={16} className="text-[var(--color-success)] shrink-0" />
              ) : uf.status === 'fail' ? (
                <AlertCircle size={16} className="text-[var(--color-danger)] shrink-0" />
              ) : (
                <Loader2 size={16} className="animate-spin text-[var(--color-accent)] shrink-0" />
              )}
              <span className="text-sm text-[var(--color-text)] flex-1 truncate">{uf.file.name}</span>
              {uf.status === 'fail' && uf.error ? (
                <span className="text-xs text-[var(--color-danger)] truncate max-w-[240px]">{uf.error}</span>
              ) : (
                <span className="text-xs text-[var(--color-text-tertiary)]">{formatSize(uf.file.size)}</span>
              )}
              {uf.status === 'uploading' && (
                <div className="w-24 h-1.5 rounded-full bg-[var(--color-bg-tertiary)] overflow-hidden">
                  <div className="h-full bg-[var(--color-action)] rounded-full transition-all" style={{ width: `${uf.progress}%` }} />
                </div>
              )}
            </div>
          ))}
          {uploadDone === uploadTotal && uploadDone > 0 && uploadFiles.every((file) => file.status === 'success') && (
            <p className="text-xs text-[var(--color-success)] text-center">{t('knowledge.success')}</p>
          )}
        </div>
      )}

      <div className="mt-8">
        <div className="flex justify-between mb-4">
          <h2 className="text-sm font-medium text-[var(--color-text)]">{t('knowledge.title')} ({docs.length})</h2>
          <button className="text-xs text-[var(--color-accent)]" onClick={() => loadDocs()}>刷新状态</button>
        </div>
        {processing && <p className="text-xs text-[var(--color-text-secondary)] mb-3">后台正在处理文档，每 10 秒更新状态；完成后即可检索。</p>}

        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-[var(--color-bg-tertiary)] rounded-lg animate-pulse" />
            ))}
          </div>
        ) : docs.length === 0 ? (
          <EmptyState icon={<FileText size={48} />} message={t('knowledge.empty')} />
        ) : (
          <div className="space-y-2">
            {docs.map((doc) => (
              <div
                key={doc.id}
                onClick={() => { if (doc.chunk_count > 0) setDetailFilename(doc.filename) }}
                className="flex items-center justify-between px-4 py-3 rounded-lg bg-[var(--color-card)] border border-[var(--color-border)] hover:border-[var(--color-accent)] cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <FileText size={16} className="text-[var(--color-text-tertiary)] shrink-0" />
                  <div className="min-w-0">
                    <p className="text-sm text-[var(--color-text)] truncate">{doc.filename}</p>
                    <p className="text-xs text-[var(--color-text-tertiary)]">
                      {{ pending: '索引排队中', running: '正在构建索引', completed: doc.chunk_count ? '可检索' : '索引完成，无有效正文', failed: '索引失败', unknown: '状态未知，请重新上传' }[doc.status] || '状态未知'}
                      {` · ${doc.chunk_count} 个切片 · ${formatDate(doc.created_at)}`}
                    </p>
                    {doc.error_message && <p className="text-xs text-[var(--color-danger)]">{doc.error_message}</p>}
                    {doc.graph_status && doc.graph_status !== 'unknown' && <p className="text-xs text-[var(--color-text-tertiary)]">
                      图谱（增强功能）：{{ waiting: '等待索引完成', pending: '后台排队中', running: '后台构建中', completed: '已完成', failed: '构建失败，不影响基础检索' }[doc.graph_status] || doc.graph_status}
                    </p>}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {(doc.status === 'failed' || doc.graph_status === 'failed') && <button
                    disabled={retrying === doc.id}
                    className="text-xs text-[var(--color-accent)] disabled:opacity-50"
                    onClick={async (event) => {
                      event.stopPropagation()
                      setRetrying(doc.id)
                      try { await knowledgeApi.retry(doc.id); toast.success('已重新排队'); await loadDocs(true) }
                      catch { toast.error('重试失败，请刷新状态后再试') }
                      finally { setRetrying(null) }
                    }}
                  >{retrying === doc.id ? '提交中…' : doc.status === 'failed' ? '重试索引' : '重试图谱'}</button>}
                  <button
                    aria-label={`删除 ${doc.filename}`}
                    onClick={(e) => { e.stopPropagation(); setDeleteTarget(doc) }}
                    className="p-1.5 rounded text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] hover:bg-[var(--color-danger-bg)] transition-colors"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <ConfirmDialog open={showClean} onOpenChange={setShowClean} title={t('knowledge.cleanAll')} message={t('knowledge.cleanConfirm')} variant="danger" confirmText={t('knowledge.cleanAll')} onConfirm={handleCleanAll} />
      <ConfirmDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)} title={t('common.confirm')} message={t('knowledge.deleteConfirm')} variant="danger" confirmText={t('note.delete')} onConfirm={handleDelete} />
      <DocumentDetailDrawer filename={detailFilename} onClose={() => setDetailFilename(null)} />
    </div>
  )
}
