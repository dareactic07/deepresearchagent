from typing import TypedDict, Annotated, List, Dict, Any, Optional

def merge_dict_list(a: Dict[str, list], b: Dict[str, list]) -> Dict[str, list]:
    if not a:
        a = {}
    c = a.copy()
    if not b:
        return c
    for k, v in b.items():
        if k in c:
            c[k] = c[k] + v
        else:
            c[k] = v
    return c

class ResearchState(TypedDict):
    topic: str
    questions: List[str]
    urls_per_question: Annotated[Dict[str, List[str]], merge_dict_list]
    extracted_content: Annotated[Dict[str, List[Dict[str, str]]], merge_dict_list]
    validated_facts: Annotated[Dict[str, List[Dict[str, Any]]], merge_dict_list]
    scores: Annotated[Dict[str, List[float]], merge_dict_list]
    research_plan: Optional[str]
    approved: bool
    final_report: Optional[str]
