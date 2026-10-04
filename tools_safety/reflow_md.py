"""Join hard-wrapped paragraph lines in a Markdown file.

md2tex turns every source newline inside a paragraph into a line break, which
makes Chinese reports look ragged.  This helper re-joins continuation lines
(without inserting a space between two CJK characters) while leaving headings,
lists, tables, code fences, block quotes, images and display math untouched.
"""
import re
import sys

STRUCT = re.compile(r"^\s*(#{1,6}\s|\||[-*+]\s|\d+\.\s|>|`|!\[|\$\$|\\\[|\s{2,}\S)")


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def join(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    if is_cjk(a[-1]) or is_cjk(b[0]):
        return a + b
    return a + " " + b


def reflow(text: str) -> str:
    out, buf, in_code = [], "", False
    for raw in text.split("\n"):
        line = raw.rstrip()
        if line.startswith("```"):
            if buf:
                out.append(buf); buf = ""
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if not line.strip():
            if buf:
                out.append(buf); buf = ""
            out.append("")
            continue
        if STRUCT.match(line):
            if buf:
                out.append(buf); buf = ""
            out.append(line)
            continue
        buf = join(buf, line.strip())
    if buf:
        out.append(buf)
    return "\n".join(out)


def main() -> None:
    path = sys.argv[1]
    src = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8", newline="\n").write(reflow(src))
    print("reflowed", path)


if __name__ == "__main__":
    main()
