export interface ApiResponse<T = unknown> {
  code: number
  message: string
  data: T
}

export interface UserInfo {
  id?: string
  user_id?: string
  uuid?: string
  username: string
  email: string
  phone?: string
  gender?: string
  bio?: string
  avatar?: string
  date_joined?: string
  is_active?: boolean
}

export interface LoginResponse {
  token: string
  user: UserInfo
}

export interface Note {
  id: string
  user_id: string
  title: string
  content: string
  tags: string[]
  category: string
  is_pinned: boolean
  created_at: string
  updated_at: string
}

export interface NoteListResponse {
  notes: Note[]
  total_count: number
}

export interface NoteTemplate {
  id: string
  user_id: string
  name: string
  icon: string
  category: string
  title: string
  content: string
  tags: string[]
  is_default: boolean
  created_at: string
  updated_at: string
}

export interface NoteStats {
  total: number
  categories: { category: string; count: number }[]
  uncategorized: number
}

export interface DeleteCategoryResponse {
  deleted_count: number
}

export interface ChatSession {
  id: string
  user_id?: string
  title: string
  metadata?: Record<string, unknown>
  created_at: string | null
  updated_at: string | null
  pending_run_id?: string | null
}

export interface ChatMessage {
  id: number
  session_id: string
  role: 'user' | 'assistant'
  content: string
  metadata?: Record<string, unknown>
  created_at: string
}

export interface KnowledgeDocument {
  id: string
  user_id: string
  filename: string
  status: string
  graph_status?: string
  graph_attempts?: number
  graph_error_message?: string | null
  attempts: number
  error_message?: string | null
  chunk_count: number
  created_at: string | null
}

export interface KnowledgeChunk {
  chunk_id: string
  index: number
  content: string
  page: number | null
  images: string[]
}

export interface KnowledgeDocumentDetail {
  id: string
  user_id: string
  md5: string
  filename: string
  chunk_count: number
  content: string
  images: string[]
  created_at: string | null
  chunks: KnowledgeChunk[]
}

export interface RelatedFragment {
  id: string
  title: string
  content_preview: string
  content: string
  similarity: number
  source: 'knowledge_base' | 'note'
}

export interface BatchIdsRequest {
  ids: string[]
}

export interface BatchCategoryRequest {
  ids: string[]
  category: string
}

export interface ReviewItem {
  review_id: string
  note_id: string
  title: string
  content_preview: string
  tags: string[]
  category: string
  review_count: number
  last_reviewed_at: string | null
  interval_days: number
}

export interface ReviewQuestion {
  question: string
  choices: string[]
  answer: string
}

export interface ReviewListData {
  reviews: ReviewItem[]
  total_count: number
}

export interface ApprovalActionRequest {
  name: string
  args: Record<string, unknown>
  description?: string
}

export interface ApprovalReviewConfig {
  action_name: string
  allowed_decisions: ('approve' | 'reject')[]
}

export interface ApprovalPayload {
  action_requests: ApprovalActionRequest[]
  review_configs: ApprovalReviewConfig[]
}

export interface AnswerSource {
  number: number
  id: string
  source: 'note' | 'knowledge_base' | 'graph' | 'web'
  title: string
  content: string
  source_id?: string | null
  page?: number | null
  chunk_index?: number | null
  url?: string | null
}

export interface SSEMessage {
  type: 'thinking' | 'response' | 'done' | 'error' | 'interrupt' | 'sources'
  sources?: AnswerSource[]
  content?: string
  session_id?: string
  stage?: string
  details?: Record<string, unknown>
  run_id?: string
  payload?: ApprovalPayload
}

export interface KnowledgeSSEMessage {
  event_type: 'start' | 'processing' | 'slicing_completed' | 'writing' | 'completed' | 'error' | 'finish'
  file_index?: number
  step?: string
  filename?: string
  progress?: number
  current?: number
  total?: number
  message?: string
  md5?: string
  knowledge_id?: string
  status?: string
  error_message?: string
}
