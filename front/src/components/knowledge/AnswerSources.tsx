import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { AnswerSource } from '../../types/api'
import DocumentDetailDrawer from './DocumentDetailDrawer'

const labels = { note: '笔记', knowledge_base: '文档', graph: '图谱', web: '网页' }

export default function AnswerSources({ sources }: { sources: AnswerSource[] }) {
  const [selectedSource, setSelectedSource] = useState<AnswerSource | null>(null)
  if (!sources.length) return null
  return (
    <div className="mt-4 space-y-2 text-xs text-[var(--color-text-secondary)]">
      <p className="font-medium">检索参考资料 · {sources.length} 条</p>
      {sources.map((source) => (
        <details key={`${source.number}-${source.id}`} className="rounded-lg border border-[var(--color-border)] p-3">
          <summary className="cursor-pointer">[{source.number}] {labels[source.source]} · {source.title}</summary>
          <p className="whitespace-pre-wrap leading-relaxed my-3">{source.content}</p>
          <div className="flex gap-3 items-center text-[var(--color-accent)]">
            {source.source === 'knowledge_base' && (
              <button onClick={() => setSelectedSource(source)}>查看原文位置</button>
            )}
            {source.source === 'note' && source.source_id && (
              <Link to={`/notes/${encodeURIComponent(source.source_id)}`}>打开原笔记</Link>
            )}
            {source.source === 'graph' && <Link to="/graph">打开知识图谱</Link>}
            {source.url && /^https?:\/\//i.test(source.url) && (
              <a href={source.url} target="_blank" rel="noopener noreferrer">打开网页</a>
            )}
            {source.page != null && source.page > 0 && <span>第 {source.page} 页</span>}
            {source.chunk_index != null && <span>切片：{source.chunk_index + 1}</span>}
          </div>
        </details>
      ))}
      <p className="text-[var(--color-text-tertiary)]">展示本轮检索到的片段，可展开核对；不表示每条资料都被回答采用。</p>
      <DocumentDetailDrawer filename={selectedSource?.title ?? null} reference={selectedSource} onClose={() => setSelectedSource(null)} />
    </div>
  )
}
