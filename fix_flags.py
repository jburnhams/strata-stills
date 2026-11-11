import re

with open("src/stratastills/db/flags.py", "r") as f:
    content = f.read()

content = re.sub(r'(s\.add\(ReportFlag\(report=report, pic=pic, \*\*vals\)\))', r'\1  # type: ignore', content)

with open("src/stratastills/db/flags.py", "w") as f:
    f.write(content)
