# fix_rft_format.py
import json, sys
sys.path.insert(0, '.')
import gsm8k_eval

rows = [json.loads(l) for l in open('data/gsm8k_rft.jsonl', encoding='utf-8')]
fixed = 0
for r in rows:
    c = r['completion'][0]['content']
    if '\\boxed' not in c:
        ans = gsm8k_eval.extract_answer(c) or ''
        r['completion'][0]['content'] = c.rstrip() + f"\n\\boxed{{{ans}}}"
        fixed += 1
with open('data/gsm8k_rft.jsonl', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'补齐 boxed: {fixed}/{len(rows)}')
