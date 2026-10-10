python3 - <<'EOF'
import json
path = "/home/smbu/.claude/projects/-home-smbu-ComfyUI/f43b36ac-4aa8-4362-a40f-f8ec00dd26be.jsonl"
hits = []
with open(path) as f:
    for line in f:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    inp = block.get("input", {})
                    cmd = inp.get("command", "")
                    if "mode-coverage-vs-frequency" in cmd and "axvspan" in cmd:
                        hits.append(cmd)
print(len(hits), "matches")
if hits:
    print(hits[-1][:200])
EOF