"""Adds a plain filled home icon (link to index.html of visualisations/) at the left of the title (first h1) of every
page of visualisations/, replacing any former version of the button. Idempotent.
python add_home.py <visualisations folder>..."""
import pathlib
import re
import sys

ICON = ('<a class="home-link" href="HREF" title="Index des visualisations" aria-label="Index des visualisations">'
        '<svg viewBox="0 0 24 24" width="0.92em" height="0.92em" aria-hidden="true">'
        '<path d="M12 2.8 2 11.4h2.8V21h5.4v-6.2h3.6V21h5.4v-9.6H22z" fill="currentColor"/></svg></a>')
STYLE = ('<style id="home-link-style">.home-link{display:inline-flex;vertical-align:-0.08em;margin-right:.4em;color:inherit;'
         'text-decoration:none}@media print{.home-link{display:none}}</style>')
FORMER = [re.compile(r'<a class="home-link".*?</a><style>\.home-link\{.*?</style>', re.DOTALL),  # (the fixed round button)
          re.compile(r'<a class="home-link".*?</a>', re.DOTALL), re.compile(r'<style id="home-link-style">.*?</style>', re.DOTALL)]

for root in sys.argv[1:]:
    base = pathlib.Path(root)
    for f in sorted(base.rglob("*.html")):
        if f.name == "index.html":
            continue
        before = f.read_text()
        s = before
        for former in FORMER:
            s = former.sub("", s)
        href = "../" * (len(f.relative_to(base).parts) - 1) + "index.html"
        body, title = re.search(r"<body[^>]*>", s), re.search(r"<h1[^>]*>", s)
        assert body and title and title.start() > body.start(), f
        s = s[:title.end()] + ICON.replace("HREF", href) + s[title.end():]
        s = s[:body.end()] + STYLE + s[body.end():]
        if s != before:
            f.write_text(s)
            print(f, "->", href)
