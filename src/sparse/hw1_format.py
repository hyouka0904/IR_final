"""HW1 postings-file format (Slide 5), copied verbatim from team27 hw1_indexer.py (IR HW1, team 27).

Copied rather than imported: hw1_indexer.py imports lxml and nltk at module level.
Record: term, df: <doc1, tf1: <pos1, pos2, ...>; doc2, tf2: <...>>
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PostingEntry:
    """One document's entry in a term's postings list: doc_id, term frequency (tf), positions, skip pointer."""
    doc_id: int
    tf: int
    positions: List[int]
    # skip pointer (index of the target entry in the postings list, its doc_id); None = no skip here
    skip: Optional[Tuple[int, int]] = None



def compute_skip_pointers(postings: List[PostingEntry]) -> List[PostingEntry]:
    """Attach skip pointers every floor(sqrt(L)) entries, L = len(postings); none when L < 4 (Slide 7).
    Entry i (i a multiple of the step) points to entry i + step. Existing pointers are reset."""
    L = len(postings)
    if L < 4:
        for p in postings:
            p.skip = None
        return postings
    step = math.isqrt(L)
    for i in range(L):
        if i % step == 0 and i + step < L:
            postings[i].skip = (i + step, postings[i + step].doc_id)
        else:
            postings[i].skip = None
    return postings


def format_posting_standard(term: str, postings: List[PostingEntry], multiline: bool = False) -> str:
    """Format one postings record as in Slide 5 (textbook Fig. 2.11, Sec. 2.4.2):
        term, df: <doc1, tf1: <pos1, pos2, ...>; doc2, tf2: <pos1, pos2, ...>>
    multiline=False (default) writes the whole record on one line, so a file can be read line by line and
    a byte offset identifies a term; multiline=True breaks lines after "df:" and after each ";" as on the slide.
    """
    df = len(postings)
    doc_parts = []
    for p in postings:
        pos_str = ", ".join(str(pos) for pos in p.positions)
        doc_parts.append(f"{p.doc_id}, {p.tf}: <{pos_str}>")
    if multiline:
        sep = ";\n"
        return f"{term}, {df}:\n<{sep.join(doc_parts)}>"
    return f"{term}, {df}: <{'; '.join(doc_parts)}>"


def parse_posting_line(line: str, attach_skips: bool = True) -> Tuple[str, int, List[PostingEntry]]:
    """Parse one postings.txt record (single-line or multi-line) back into (term, df, [PostingEntry, ...]).
    Skip pointers are not stored in the file; they are recomputed here when attach_skips=True."""
    raw = (
        line.strip()
        .replace(":\r\n<", ": <")
        .replace(":\n<", ": <")
        .replace(";\r\n", "; ")
        .replace(";\n", "; ")
    )
    head_part, body_part = raw.split(": <", 1)
    term, df_str = head_part.rsplit(", ", 1)
    df = int(df_str)
    # drop the record's closing '>'
    body = body_part[:-1]
    postings: List[PostingEntry] = []
    for chunk in body.split("; "):
        doc_tf_part, pos_part = chunk.split(": <", 1)
        doc_id_str, tf_str = doc_tf_part.split(", ", 1)
        positions = [int(x) for x in pos_part[:-1].split(", ")]
        postings.append(PostingEntry(doc_id=int(doc_id_str), tf=int(tf_str), positions=positions))
    if attach_skips:
        compute_skip_pointers(postings)
    return term, df, postings

