from typing import List, Any


class Edit:
    def __init__(self, orig_text: str, correct_text: str, begin: int, end: int, additional: Any = None):
        self.orig_text = orig_text
        self.correct_text = correct_text
        self.begin = begin
        self.end = end
        self.additional = additional

    def __str__(self) -> str:
        return f'{self.orig_text}[{self.begin},{self.end}] -> {self.correct_text}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        return self.begin == other.begin and self.end == other.end and self.correct_text == other.correct_text


class Sentence:
    def __init__(self, sent: str, doc_offset: int = 0):
        self.sent = sent
        self.doc_offset = doc_offset

    def set_edits(self, edits: List[Edit]):
        self.edits = edits

    def rebuild_text(self, show_diff=False):
        if self.edits is None:
            return self.sent
        out = []
        i = 0
        for edit in self.edits:
            out.append(self.sent[i: edit.begin])
            if show_diff:
                rpl = f'[{edit.orig_text}->{edit.correct_text}]'
            else:
                rpl = edit.correct_text
            out.append(rpl)
            i = edit.end
        out.append(self.sent[i:])
        return ''.join(out)
