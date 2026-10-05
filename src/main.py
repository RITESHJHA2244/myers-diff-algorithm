"""Assignment 1 - Myers' O(ND) diff.

    python src/main.py lines     A B   -> minimal line diff of A -> B   (Part A)
    python src/main.py highlight A B   -> line diff + changed characters (Part B)

Vocabulary used in the comments (same as Myers' 1986 paper)
-----------------------------------------------------------
  x, y   : x = how many items of A are consumed, y = how many items of B.
           The edit graph is a grid; moving RIGHT (x+1) = delete a[x],
           moving DOWN (y+1) = insert b[y], moving DIAGONALLY = keep (a[x]==b[y]).
  k      : diagonal number, k = x - y.
  d      : number of edits (deletes + inserts) used so far.
  V      : V[k] = furthest x reached on diagonal k using exactly d edits.
  snake  : a run of free diagonal moves (matching items) after one edit.

"""

import sys


# ---------------------------------------------------------------------------
# 1. Reading files
# ---------------------------------------------------------------------------
def read_lines(path):
    """Read a file as RAW BYTES and split it into lines (as bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")      # split only on the byte 0x0A; b"\r" stays in the line
    if lines and lines[-1] == b"":  # last piece empty -> final newline / empty file
        lines.pop()
    return lines                    # b"" -> [] ; b"\n" -> [b""] ; b"a\r\nb\r\n" -> [b"a\r", b"b\r"]


# ---------------------------------------------------------------------------
# 2. Myers' algorithm on ANY two sequences (lines as ints, or characters)
#
#    Result: two bytearrays.  a_del[i] == 1  -> a[i] is deleted
#                             b_ins[j] == 1  -> b[j] is inserted
#    Everything unmarked is "kept" (the kept items match, in order).
#
#    Memory-friendly version of Myers: instead of storing the V array of every
#    d (O(D^2) memory) we use the paper's "middle snake" idea (section 4b):
#    search from the start AND from the end at the same time until the two
#    searches meet; the meeting snake is part of an optimal path, so we only
#    have to solve the left part and the right part recursively.  Memory O(N+M).
# ---------------------------------------------------------------------------
def myers_mark(a, b):
    """Return (a_del, b_ins) marking a minimal edit script from a to b."""
    a_del = bytearray(len(a))
    b_ins = bytearray(len(b))
    _solve(a, b, 0, len(a), 0, len(b), a_del, b_ins)
    return a_del, b_ins


def _solve(a, b, alo, ahi, blo, bhi, a_del, b_ins):
    """Diff a[alo:ahi] against b[blo:bhi] (no copying, only index bounds)."""
    # Free snakes at both ends cost zero edits: skip them.
    while alo < ahi and blo < bhi and a[alo] == b[blo]:
        alo += 1
        blo += 1
    while alo < ahi and blo < bhi and a[ahi - 1] == b[bhi - 1]:
        ahi -= 1
        bhi -= 1

    if alo == ahi:                      # nothing left in A: everything in B is an insert
        if blo < bhi:
            b_ins[blo:bhi] = b"\x01" * (bhi - blo)
        return
    if blo == bhi:                      # nothing left in B: everything in A is a delete
        a_del[alo:ahi] = b"\x01" * (ahi - alo)
        return

    # Here D >= 2, so the middle snake splits the problem into two smaller ones.
    xs, ys, xe, ye = _middle_snake(a, b, alo, ahi, blo, bhi)
    _solve(a, b, alo, alo + xs, blo, blo + ys, a_del, b_ins)      # before the snake
    # the snake (xs,ys)->(xe,ye) is all matches = keeps, nothing to mark
    _solve(a, b, alo + xe, ahi, blo + ye, bhi, a_del, b_ins)      # after the snake


def _middle_snake(a, b, alo, ahi, blo, bhi):
    """Find the middle snake of a[alo:ahi] vs b[blo:bhi].

    Runs the forward search (from top-left) and the backward search (from
    bottom-right) one d at a time until they overlap.
    Returns (xs, ys, xe, ye): snake start and end, relative to (alo, blo).
    """
    sa = a[alo:ahi]                     # the two sub-sequences (one copy per call)
    sb = b[blo:bhi]
    ra = sa[::-1]                       # reversed copies for the backward search
    rb = sb[::-1]
    n = len(sa)
    m = len(sb)
    delta = n - m                       # diagonal of the bottom-right corner
    odd = delta & 1
    max_d = (n + m + 1) // 2
    # V is indexed directly by k.  k can be negative: Python lists wrap negative
    # indices around to the end, and the list is long enough (2*max_d+3) that
    # the diagonals -d-1 .. d+1 never collide.
    vf = [0] * (2 * max_d + 3)          # forward V : furthest x on diagonal k
    vb = [0] * (2 * max_d + 3)          # backward V: same, on the reversed sequences

    for d in range(max_d + 1):
        # ---------------- forward search with d edits ----------------
        for k in range(-d, d + 1, 2):
            # Edit decision: come from diagonal k+1 (an INSERT: move down, x same)
            # or from diagonal k-1 (a DELETE: move right, x+1). Take the one that got further.
            if k == -d or (k != d and vf[k - 1] < vf[k + 1]):
                x = vf[k + 1]                   # insert
            else:
                x = vf[k - 1] + 1               # delete
            y = x - k
            x0 = x
            y0 = y
            # follow the snake: free diagonal moves while the items are equal
            while x < n and y < m and sa[x] == sb[y]:
                x += 1
                y += 1
            vf[k] = x
            # If delta is odd the paths can first overlap during the forward pass.
            if odd and delta - (d - 1) <= k <= delta + (d - 1):
                # reverse diagonal of k is delta-k ; has the backward path passed this point?
                if x + vb[delta - k] >= n:
                    return x0, y0, x, y         # this forward snake is the middle snake

        # ---------------- backward search with d edits ----------------
        # Same thing on the reversed sequences: x,y count items taken from the end.
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and vb[k - 1] < vb[k + 1]):
                x = vb[k + 1]
            else:
                x = vb[k - 1] + 1
            y = x - k
            x0 = x
            y0 = y
            while x < n and y < m and ra[x] == rb[y]:
                x += 1
                y += 1
            vb[k] = x
            # If delta is even the paths first overlap during the backward pass.
            if (not odd) and delta - d <= k <= delta + d:
                if vf[delta - k] + x >= n:
                    # convert reversed coordinates back to forward ones
                    return n - x, m - y, n - x0, m - y0

    raise AssertionError("middle snake not found")  # cannot happen


# ---------------------------------------------------------------------------
# 3. Part A: line diff
# ---------------------------------------------------------------------------
def diff_lines(a_lines, b_lines):
    """Return (a_del, b_ins) for the two lists of byte-lines (minimal diff).

    Speed-up that keeps minimality: lines are mapped to small ints, and lines
    that occur in only ONE file can never be kept by any diff, so they are
    marked deleted/inserted directly and removed before running Myers.
    """
    ids = {}
    a_ids = [ids.setdefault(line, len(ids)) for line in a_lines]
    b_ids = [ids.setdefault(line, len(ids)) for line in b_lines]
    in_a = set(a_ids)
    in_b = set(b_ids)

    a_del = bytearray(len(a_ids))
    b_ins = bytearray(len(b_ids))
    a_keep_idx = []                     # original indices of lines that exist in both files
    b_keep_idx = []
    fa = []
    fb = []
    for i, v in enumerate(a_ids):
        if v in in_b:
            a_keep_idx.append(i)
            fa.append(v)
        else:
            a_del[i] = 1                # only in A -> must be deleted
    for j, v in enumerate(b_ids):
        if v in in_a:
            b_keep_idx.append(j)
            fb.append(v)
        else:
            b_ins[j] = 1                # only in B -> must be inserted

    sub_del, sub_ins = myers_mark(fa, fb)
    for i, flag in enumerate(sub_del):
        if flag:
            a_del[a_keep_idx[i]] = 1
    for j, flag in enumerate(sub_ins):
        if flag:
            b_ins[b_keep_idx[j]] = 1
    return a_del, b_ins


def build_script(a_del, b_ins):
    """Walk both files and build the edit script.

    Each item is either (i, j) for a keep line, or ([del idx], [ins idx]) for
    a change block.

    DELETE-FIRST RULE: a change block is collected completely (all deleted
    A-lines, all inserted B-lines) and always emitted as deletes, then inserts.
    """
    n = len(a_del)
    m = len(b_ins)
    i = j = 0
    script = []
    while i < n or j < m:
        dels = []
        ins = []
        while i < n and a_del[i]:
            dels.append(i)
            i += 1
        while j < m and b_ins[j]:
            ins.append(j)
            j += 1
        if dels or ins:
            script.append((dels, ins))
        else:                           # both unmarked: a[i] == b[j], a keep line
            script.append((i, j))
            i += 1
            j += 1
    return script


# ---------------------------------------------------------------------------
# 4. Part B: character ranges
# ---------------------------------------------------------------------------
def marks_to_ranges(marks):
    """Turn a 0/1 bytearray into 'start-end,start-end' (end exclusive) or '.'.

    Consecutive marked characters form ONE run, so touching ranges are
    merged automatically, ranges are sorted and never overlap.
    """
    parts = []
    i = 0
    n = len(marks)
    while i < n:
        if marks[i]:
            start = i
            while i < n and marks[i]:
                i += 1
            parts.append("%d-%d" % (start, i))
        else:
            i += 1
    return ",".join(parts) if parts else "."


def highlight_pair(old_bytes, new_bytes):
    """Return the '? old | new' line (as bytes) for one paired - / + line."""
    # Unicode: decode to str, so indexing/iteration is by CODE POINT, not by byte.
    # (Part B files are valid UTF-8; surrogateescape only guards against surprises.)
    old = old_bytes.decode("utf-8", "surrogateescape")
    new = new_bytes.decode("utf-8", "surrogateescape")
    old_marks, new_marks = myers_mark(old, new)     # same Myers, items = characters
    text = "? %s | %s" % (marks_to_ranges(old_marks), marks_to_ranges(new_marks))
    return text.encode("ascii")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) != 4 or argv[1] not in ("lines", "highlight"):
        sys.stderr.write("usage: main.py lines|highlight A B\n")
        return 2
    mode, path_a, path_b = argv[1], argv[2], argv[3]

    try:                                # read BOTH files before printing anything
        a_lines = read_lines(path_a)
        b_lines = read_lines(path_b)
    except OSError as err:
        sys.stderr.write("error: cannot read input file: %s\n" % err)
        return 2

    a_del, b_ins = diff_lines(a_lines, b_lines)
    script = build_script(a_del, b_ins)

    out = []
    for item in script:
        first, second = item
        if isinstance(first, int):                      # keep line
            out.append(b" " + a_lines[first] + b"\n")
            continue
        for i in first:                                 # all deletes first ...
            out.append(b"-" + a_lines[i] + b"\n")
        for idx, j in enumerate(second):                # ... then all inserts
            out.append(b"+" + b_lines[j] + b"\n")
            # Part B: pair 1st '-' with 1st '+', 2nd with 2nd, ...
            # leftovers (idx >= number of deletes) are unpaired -> no '?' line.
            if mode == "highlight" and idx < len(first):
                out.append(highlight_pair(a_lines[first[idx]], b_lines[j]) + b"\n")

    sys.stdout.buffer.write(b"".join(out))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))