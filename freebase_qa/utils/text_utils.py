import re
from typing import List
from typing import Tuple

class TextUtils:
    @staticmethod
    def extract_answer(text: str) -> str:
        """从文本中提取答案"""
        start = text.find("{")
        end = text.find("}")
        return text[start+1:end].strip() if start != -1 and end != -1 else ""
    
    @staticmethod
    def is_true(text: str) -> bool:
        """检查文本是否表示肯定"""
        return text.lower().strip().replace(" ", "") == "yes"
    
    @staticmethod
    def is_yes_in_response(response: str) -> bool:
        """判断响应中是否出现独立的 'Yes'（区分大小写）"""
        if not response:
            return False
        
        # 使用正则匹配独立的 "Yes"
        # \b 表示单词边界，确保匹配的是完整单词
        return bool(re.search(r'\bYes\b', response))

    @staticmethod
    def check_finish(entities: List[str]) -> Tuple[bool, List[str]]:
        """检查实体列表是否完成"""
        if all(e == "[FINISH_ID]" for e in entities):
            return True, []
        return False, [e for e in entities if e != "[FINISH_ID]"]
    
    @staticmethod
    def filter_unknown_entities(entity_candidates: List[str], entity_candidates_id: List[int]) -> Tuple[List[str], List[int]]:
        """同时删除名称和对应的 ID 中那些标记为 "UnName_Entity" 的条目。"""
        filtered_names, filtered_ids = [], []
        for name, eid in zip(entity_candidates, entity_candidates_id):
            if name != "UnName_Entity":
                filtered_names.append(name)
                filtered_ids.append(eid)
        # 特殊情况：如果仅有一个，并且就是 “UnName_Entity”，保留它
        if not filtered_names and len(entity_candidates) == 1:
            return entity_candidates, entity_candidates_id
        return filtered_names, filtered_ids
    
    @staticmethod
    def extract_json(text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON object found in text")