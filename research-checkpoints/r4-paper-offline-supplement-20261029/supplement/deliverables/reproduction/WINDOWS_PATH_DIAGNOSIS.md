# Portable-reproduction environment diagnosis

The initial isolated restoration passed all 299 tests and all 404 replays. A later
fresh copy under a deeper delivery/portable-smoke path passed 298 tests but failed
one object-store filesystem test: its nested pytest temporary path plus user hash
and 64-character content digest exceeded the ordinary Windows path-length limit.
The failure was FileNotFoundError while writing a test object, not a label,
extractor, scorer or application-state disagreement.

The supplemental reproduce_baseline.py now passes --basetemp <NEW-workspace>/t.
It refuses an existing temporary tree or prior reproduction log, preserving user
files. This changes only test temporary-file location; no published test,
application, parser, prompt, gold, output or scoring code is changed.

The first failed portability logs are retained as portable-*-first*.log. The final
fresh-copy reproduction receipt records the rerun. The server already has tzdata
for Asia/Karachi, and both bundled modified test files are present and unchanged.
'''
