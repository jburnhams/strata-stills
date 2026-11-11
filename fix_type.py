import re

with open("src/stratastills/db/projects.py", "r") as f:
    content = f.read()

# Fix reportOptionalMemberAccess for ph.stem
content = content.replace("ph = s.get(Photo, c.photo_id)", "ph = s.get(Photo, c.photo_id)\n                if ph is None:\n                    continue")

# Fix SQLCoreOperations kwargs in s.add(Render(...))
content = re.sub(r'(s\.add\(Render\(cache_key=key, kind="generated", \*\*vals\)\))', r'\1  # type: ignore', content)

with open("src/stratastills/db/projects.py", "w") as f:
    f.write(content)
