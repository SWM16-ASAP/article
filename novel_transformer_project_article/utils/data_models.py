from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class FolderData:
    """
    모든 콘텐츠 유형(literature, article 등)이 파싱된 후
    공통적으로 사용하게 될 표준 데이터 구조.
    워크플로우의 각 노드에 전달되는 데이터의 형태.
    """
    content_type: str
    full_text: str
    title: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))