from enum import StrEnum, auto

from pydantic import BaseModel


class SkillArea(StrEnum):
    PITCH = auto()
    TIMING = auto()


class FrontMatter(BaseModel):
    # Document Level
    title: str
    source_url: str
    license: str
    skill_area: SkillArea
    content_type: str
    date_collected: str

    # Optional
    author: str | None = None
    level: str | None = None
    tags: list[str] = []
    notes: str | None = None


class Payload(BaseModel):
    frontmatter: FrontMatter
    # Chunk level
    text: str
    section_title: str
    doc_id: str
    chunk_index: int
