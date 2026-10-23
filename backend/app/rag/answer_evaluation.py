"""Frozen answer annotations and mechanical checks, never an accuracy judge."""
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation import EvalQuestion, FixedDataset, answer_checks


class SupportSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    point: str = Field(min_length=1)


class AnswerQuestion(EvalQuestion):
    supporting_spans: list[SupportSpan]
    challenge_tags: list[str] = Field(default_factory=list)


class AnswerDataset(FixedDataset):
    questions: list[AnswerQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_support(self):
        documents = {doc.id: doc.text for doc in self.documents}
        for question in self.questions:
            if {span.source_id for span in question.supporting_spans} != set(question.relevant_source_ids):
                raise ValueError(f"{question.id}: supporting spans must cover exactly the relevant sources")
            for span in question.supporting_spans:
                if span.quote not in documents[span.source_id]:
                    raise ValueError(f"{question.id}: support must be a verbatim document substring")
        return self


def mechanical_checks(answer: str, sources: list[dict], question: AnswerQuestion) -> dict:
    """A real citation number proves identity only, not support for its adjacent claim."""
    numbers = [int(number) for group in re.findall(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]", answer)
               for number in re.split(r"\s*[,，]\s*", group)]
    by_number = {source.get("number"): source.get("source_id") for source in sources}
    duplicates = len(by_number) != len(sources)
    required = set(question.relevant_source_ids)
    result = answer_checks(answer, question)
    result.update({
        "citation_numbers": numbers,
        "has_citations": bool(numbers),
        "citation_numbers_valid": not duplicates and all(number in by_number for number in numbers),
        "all_required_sources_returned": required <= set(by_number.values()) if required else None,
        "all_required_sources_cited": required <= {by_number.get(n) for n in numbers} if required else None,
        "semantic_support": "requires_review",
    })
    return result
