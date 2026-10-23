import type { AnswerSource, KnowledgeDocumentDetail } from '../types/api'

export type SourceReference = Pick<AnswerSource, 'id' | 'source_id' | 'content'>

// 历史快照只能在当前来源身份、切片 ID 和已有正文都相符时定位。
export function locateSource(detail: KnowledgeDocumentDetail, reference?: SourceReference | null) {
  if (!reference) return { status: 'none' as const, chunk: null }
  if (reference.source_id && reference.source_id !== detail.id) {
    return { status: 'changed' as const, chunk: null }
  }
  const chunk = detail.chunks.find((item) => item.chunk_id === reference.id)
  if (!chunk) return { status: 'missing' as const, chunk: null }
  const normalize = (text: string) => text.replace(/\s+/gu, ' ').trim()
  const snapshot = normalize(reference.content)
  if (!snapshot || !normalize(chunk.content).includes(snapshot)) {
    return { status: 'changed' as const, chunk: null }
  }
  return { status: 'matched' as const, chunk }
}
