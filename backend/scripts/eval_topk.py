"""Compare frozen local-document evidence prefixes through production RAG guards.

Read-only corpus access; no chat/session/index writes. This is a component
experiment, not a test of the planner, tools, conversations, web or HTTP routes.
Prepare once, run dev, manually review and lock a choice, then run validation.
"""
import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

BACKEND = Path(__file__).resolve().parent.parent
if not (BACKEND / 'app').exists():
    BACKEND = Path('/app')
sys.path.insert(0, str(BACKEND))

from app.core.background_init import init_manager  # noqa: E402
from app.core.settings import settings  # noqa: E402
from app.db.db_config import AsyncSessionLocal, async_engine  # noqa: E402
from app.graph.storage.neo4j_client import close_neo4j_driver  # noqa: E402
from app.graph.storage.neo4j_graph_store import Neo4jGraphStore  # noqa: E402
from app.rag.agentic_rag.evaluator import AnswerabilityEvaluator  # noqa: E402
from app.rag.agentic_rag.evidence import prepare_evidence_context  # noqa: E402
from app.rag.agentic_rag.local_retriever import LocalRetriever  # noqa: E402
from app.rag.agentic_rag.planner import _resolve_shared_chat_model  # noqa: E402
from app.rag.agentic_rag.reranker import MAX_CANDIDATES, EvidenceReranker  # noqa: E402
from app.rag.agentic_rag.schemas import Evidence, RetrievalStep  # noqa: E402
from app.rag.agentic_rag.sources import public_sources  # noqa: E402
from app.rag.answer_verifier import AnswerVerifier, _resolve_review_model  # noqa: E402
from app.utils.factory import EmbedModelFactory, resolve_embed_config  # noqa: E402
from app.utils.prompt_loader import load_prompt  # noqa: E402


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def norm(value):
    return re.sub(r'\s+', '', value).lower()


async def corpus_snapshot(uid):
    async with AsyncSessionLocal() as db:
        docs = [dict(r) for r in (await db.execute(text(
            'SELECT * FROM graph_docs WHERE user_id=:uid ORDER BY id'), {'uid':uid})).mappings()]
    chunks = [dict(r) for r in await Neo4jGraphStore()._run(
        'MATCH (c:Chunk {user_id:$uid}) RETURN c.id AS id, c.source_id AS source_id, '
        'c.text AS text, c.chunk_index AS chunk_index, c.page AS page, c.kind AS kind ORDER BY id', {'uid':uid})]
    return {'sha256':digest({'docs':docs,'chunks':chunks}), 'document_count':len(docs),
            'document_chunks':sum(r['kind']=='doc' for r in chunks),
            'other_chunks':sum(r['kind']!='doc' for r in chunks)}


def config_snapshot():
    paths = ['app/prompt/rag_context_prompt.md','app/prompt/agentic_rag_answerability.md',
             'app/rag/answer_verifier.py','app/rag/agentic_rag/evaluator.py',
             'app/rag/agentic_rag/evidence.py','app/rag/agentic_rag/local_retriever.py',
             'app/rag/agentic_rag/reranker.py','app/graph/storage/neo4j_graph_store.py']
    return {'answer_model': settings.OPENAI_MODEL_NAME,
            'embedding_model':resolve_embed_config()['model'], 'reranker_model':settings.RERANKER_MODEL,
            'candidate_k':settings.RERANKER_CANDIDATE_K,
            'runtime_local_context_k':getattr(settings, 'RAG_CONTEXT_K', 8),
            'verification_enabled':settings.ANSWER_VERIFICATION_ENABLED,
            'answerability_timeout':settings.ANSWERABILITY_TIMEOUT_SECONDS,
            'generation_timeout':settings.AGENT_RUN_TIMEOUT_SECONDS,
            'verification_timeout':settings.ANSWER_VERIFICATION_TIMEOUT_SECONDS,
            'file_sha256':{p:hashlib.sha256((BACKEND/p).read_bytes()).hexdigest() for p in paths}}


class RecordingModel:
    """Capture provider usage, never secrets or hidden reasoning content."""
    def __init__(self, model, stage, calls):
        self.model, self.stage, self.calls = model, stage, calls

    async def ainvoke(self, messages):
        row = {'stage':self.stage,'status':'started','usage':None}
        self.calls.append(row)
        start = time.perf_counter()
        try:
            result = await self.model.ainvoke(messages)
            usage = getattr(result, 'usage_metadata', None)
            metadata = getattr(result, 'response_metadata', None) or {}
            row.update(status='completed', usage=usage,
                       provider_token_usage=metadata.get('token_usage'),
                       finish_reason=metadata.get('finish_reason'))
            return result
        except BaseException as exc:
            row.update(status='failed_or_cancelled', error_type=type(exc).__name__)
            code = getattr(exc,'status_code',None)
            if type(code) is int:
                row['http_status'] = code
            raise
        finally:
            row['elapsed_ms'] = round((time.perf_counter()-start)*1000,2)


async def prepare(args, dataset):
    target = args.output / 'frozen.json'
    if target.exists():
        raise ValueError('frozen_pool_already_exists')
    config = config_snapshot()
    if not config['verification_enabled'] or config['candidate_k'] != dataset['design']['candidate_k']:
        raise ValueError('configuration_does_not_match_design')
    before = await corpus_snapshot(args.user_id)
    init_manager.embed_model = EmbedModelFactory().generator()
    reranker = EvidenceReranker()
    if not reranker.active:
        raise ValueError('reranker_must_be_active')
    report = {'created_at':datetime.now(UTC).isoformat(), 'scope':dataset['scope'],
              'dataset_sha256':digest(dataset), 'config':config, 'corpus_before':before,
              'questions':[], 'status':'preparing',
              'shared_usage':'Embedding/rerank SDK does not expose provider token usage here; unknown. Shared once per question, never charged three times in arm totals.'}
    save(args.output/'dataset.json',dataset)
    for q in dataset['questions']:
        start = time.perf_counter()
        pool = await LocalRetriever().search(args.user_id,[RetrievalStep(
            tool='search_knowledge_base',query=q['query'],top_k=config['candidate_k'])])
        retrieval_ms = (time.perf_counter()-start)*1000
        ranked = await reranker.rank(q['query'],pool,limit=MAX_CANDIDATES)
        row = {'id':q['id'], 'retrieval_ms':round(retrieval_ms,2),'rerank':ranked.summary(),
               'ranked':[e.model_dump() for e in ranked.evidences]}
        report['questions'].append(row)
        save(args.output/'preparation.json',report)
        print(json.dumps({'phase':'freeze','id':q['id'],'candidates':len(pool), 'rerank':ranked.status}),flush=True)
        if ranked.status != 'applied' or not ranked.evidences:
            raise ValueError('failed_shared_retrieval_or_rerank')
    report['corpus_after'] = await corpus_snapshot(args.user_id)
    if report['corpus_after'] != before:
        raise ValueError('corpus_changed_during_freeze')
    report['status'] = 'frozen'
    save(target,report)


async def run_arm(q, frozen, k):
    ranked = [Evidence.model_validate(e) for e in frozen['ranked']]
    prepared = prepare_evidence_context(ranked[:k],max_chars=6000)
    sources = public_sources(prepared.evidences)
    calls, diagnostics, verification = [], {}, {}
    row = {'id':q['id'],'k':k,'context_chars':len(prepared.text),'actual_count':len(sources),
           'truncated_count':sum(bool(e.metadata.get('context_truncated')) for e in prepared.evidences),
           'sources':sources,'source_window_sha256':digest(sources),
           'gold_quote_coverage':[any(norm(s['quote']) in norm(e.content) for e in prepared.evidences)
                                  for s in q['gold_quotes']],
           'calls':calls,'status':'running','answer':'','draft':''}
    start = time.perf_counter()
    try:
        gate = await AnswerabilityEvaluator(chat_model=RecordingModel(
            _resolve_review_model(),'answerability',calls),diagnostics=diagnostics).evaluate(q['query'],prepared.evidences)
        row['gate'] = gate.model_dump()
        if gate.status == 'unavailable':
            row.update(status='gate_unavailable',answer='暂时无法完成资料核对，因此无法依据现有资料给出可靠回答。请稍后重试。')
        elif not gate.answerable:
            row.update(status='insufficient_evidence',answer='现有资料中没有足够证据回答这个问题，无法确定所问的信息。请补充相关资料，或明确需要核对的内容。')
        else:
            model = RecordingModel(_resolve_shared_chat_model(),'generation',calls)
            response = await asyncio.wait_for(model.ainvoke([
                SystemMessage(content=load_prompt('rag_context_prompt').replace('{context}',prepared.text)),
                HumanMessage(content=q['query'])]),timeout=settings.AGENT_RUN_TIMEOUT_SECONDS)
            if not isinstance(response.content,str):
                raise ValueError('generation_not_text')
            row['draft'] = response.content
            result = await AnswerVerifier(RecordingModel(_resolve_review_model(),'verification',calls)).verify(
                q['query'],row['draft'],sources,diagnostics=verification)
            row.update(status=result.status,answer=result.text,kept_blocks=result.kept_blocks,
                       removed_blocks=result.removed_blocks,repaired_citations=result.repaired_citations)
    except Exception as exc:
        row.update(status='error',error_type=type(exc).__name__)
    row['elapsed_ms'] = round((time.perf_counter()-start)*1000,2)
    row['term_checks'] = [any(norm(term) in norm(row['answer']) for term in group) for group in q['required_terms']]
    row['citations'] = sorted(set(int(n) for n in re.findall(r'\[(\d+)\]',row['answer'])))
    row['unknown_usage_calls'] = sum(c.get('usage') is None for c in calls)
    row['known_tokens'] = {key:sum((c.get('usage') or {}).get(key,0) for c in calls)
                           for key in ('input_tokens','output_tokens','total_tokens')}
    # Private trace contains only visible draft/review/source data, never provider hidden reasoning.
    row['diagnostics'] = {'answerability':diagnostics,'verification':verification}
    return row


async def run(args, dataset):
    frozen = json.loads((args.output/'frozen.json').read_text(encoding='utf-8'))
    if digest(dataset)!=frozen['dataset_sha256'] or config_snapshot()!=frozen['config']:
        raise ValueError('dataset_or_runtime_changed')
    if await corpus_snapshot(args.user_id)!=frozen['corpus_before']:
        raise ValueError('corpus_changed_since_freeze')
    if args.split=='validation':
        locked = json.loads((args.output/'choice.json').read_text(encoding='utf-8'))
        if locked.get('dataset_sha256')!=frozen['dataset_sha256'] or locked.get('k') not in (3,5,8):
            raise ValueError('lock_a_dev_choice_before_validation')
    path = args.output/f'{args.split}.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'split':args.split,'dataset_sha256':frozen['dataset_sha256'],
        'created_at':datetime.now(UTC).isoformat(),'rows':[],'status':'running'}
    done = {(r['id'],r['k']) for r in report['rows']}
    pools = {q['id']:q for q in frozen['questions']}
    for i,q in enumerate(q for q in dataset['questions'] if q['split']==args.split):
        ks = [3,5,8]
        ks = ks[i%3:]+ks[:i%3]
        for k in ks:
            if (q['id'],k) in done:
                continue
            row = await run_arm(q,pools[q['id']],k)
            report['rows'].append(row)
            save(path,report)
            print(json.dumps({'phase':args.split,'id':q['id'],'k':k,'status':row['status'],
                'elapsed_ms':row['elapsed_ms'],'known_tokens':row['known_tokens'],
                'unknown_usage_calls':row['unknown_usage_calls']}),flush=True)
    report['corpus_after'] = await corpus_snapshot(args.user_id)
    report['corpus_unchanged'] = report['corpus_after']==frozen['corpus_before']
    report['status'] = 'requires_manual_review'
    save(path,report)


async def main_async(args):
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        if args.mode=='prepare':
            await prepare(args,dataset)
        else:
            await run(args,dataset)
    finally:
        await close_neo4j_driver()
        await async_engine.dispose()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--user-id',required=True)
    parser.add_argument('--split',choices=['dev','validation'],default='dev')
    asyncio.run(main_async(parser.parse_args()))


if __name__=='__main__':
    main()
