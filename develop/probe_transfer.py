r"""다른 seq_len 로 한 번 더 트레이스한 결과(probe)를 **등가류 단위**로 옮겨 override 를 만든다.

왜 필요한가
-----------
두 config 필드가 우연히 같은 정수면 트레이스에 남은 정보만으로 두 축을 가를 수 없다.
DeepSeek-V4-Pro 가 그랬다: T=2048 에서 `d_head == T/m_csa == 512`. 같은 모델을 T=1920 으로
한 번 더 트레이스하면 d_head=512, T/m_csa=480 으로 갈린다(1920 은 m_csa=4 와 m_hca=128 의
공배수라 나머지 없이 구조가 유지된다). 그 판정을 원래 트레이스로 옮기는 것이 이 도구다.

지켜야 할 것 네 가지 -- 전부 실제로 틀려 보고 나온 것이다(2026-09-04)
-----------------------------------------------------------------
1. **shape 으로 대응시키지 않는다.** 두 실행은 shape 이 달라지는 것이 목적이다. 자리는
   (module_key, layer_type, op_type, module-local nth, field, shape_index, axis) 로 짚는다.
   두 실행의 op 수가 (module, op_type) 별로 같은지 먼저 확인하라(V4-Pro 는 46,268개로 동일).
2. **layer_type 을 키에 넣는다.** module_key 는 층 인덱스를 `.*` 로 접으므로 하이브리드
   스택에서는 층 타입이 다른 자리가 같은 칸에 겹친다. 빼먹었더니 csa 층에서 잰 값이 hca 층에
   적용돼 등가류 충돌 91건이 났다.
3. **옮기는 단위는 자리가 아니라 등가류다.** 자리 단위로 옮기면 한 등가류가 두 이름으로
   쪼개져 충돌 90건 + reshape_incons 60건이 난다. 등가류는 "이 자리들은 같은 축"이라는
   주장이고 probe 는 "그 축의 이름"에 답하므로 둘을 합치는 단위는 등가류다. 클래스마다 대표
   자리 하나만 적고 `spread: class` 로 나머지를 끌고 간다.
4. **반증을 요구한다.** probe 가 base 이름을 실제로 깨뜨린 자리가 하나는 있어야 한다. 없으면
   probe 에서도 두 이름의 값이 같다는 뜻이고, 그때 이름을 바꾸는 건 판별이 아니라 probe 실행
   라벨러의 잡음을 베끼는 것이다. V4-Pro 에서 이 조건이 486 클래스를 걸렀다(g_o 와
   T/m_hca+1 은 T=1920 에서도 둘 다 16, d_rope 와 n_h_I 는 둘 다 64).

base 는 **probe 교정이 들어가지 않은** 트레이스여야 한다. 이미 적용된 develop/out 을 base 로
쓰면 자기 자신과 비교하게 된다(처음에 그렇게 해서 386개를 헛짚었다).

실행:
    .venv\Scripts\python.exe src\run.py --profile develop\models\probe-<...>.yaml --out develop\probe_out
    .venv\Scripts\python.exe develop\probe_transfer.py
   -> develop/probe_out/<model>/class_fixes.json  (rules/label_overrides.yaml 항목으로 옮긴다)
"""
import os, sys, json, io, re, collections
sys.path.insert(0, 'src')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import axis_classes as ac, build_table as BT
from anchors import module_key

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'deepseek-ai__DeepSeek-V4-Pro'
# base 는 probe 교정이 **들어가지 않은** 트레이스여야 한다. develop/out 에 있으면 그것을,
# 없으면 승격본을 쓴다.
BASE = (f'develop/out/{MODEL}' if os.path.isdir(f'develop/out/{MODEL}/full')
        else f'models/{MODEL}')
PROBE = f'develop/probe_out/{MODEL}'
print(f'base  {BASE}')
print(f'probe {PROBE}')
LIDX = re.compile(r'\.layers\.(\d+)\.')


def sch(d):
    return (json.load(open(f'{d}/full/provenance.json', encoding='utf-8')).get('config') or {}).get('layer_types') or []


SB, SP = sch(BASE), sch(PROBE)
stP = {k: v for k, v in json.load(open(f'{PROBE}/full/provenance.json', encoding='utf-8'))['symbol_table'].items()
       if isinstance(v, int)}


def ev(e):
    try:
        return int(eval(e, {'__builtins__': {}}, dict(stP)))
    except Exception:
        return None


def load(d, ph, s):
    rows = [json.loads(l) for l in open(f'{d}/full/{ph}.trace.raw.jsonl', encoding='utf-8')]
    conc = {c['op_id']: c for c in (json.loads(l) for l in
            open(f'{d}/full/{ph}.shapes.concrete.jsonl', encoding='utf-8'))}
    ordn = ac.op_ordinals(rows)
    kp, m = {}, {}
    for r in rows:
        mp = r.get('module_path') or ''
        g = LIDX.search(mp)
        kind = s[int(g.group(1))] if g and int(g.group(1)) < len(s) else None
        k0 = (module_key(mp) or '(root)', kind, r.get('op_type'), ordn.get(r.get('op_id')))
        kp[r.get('op_id')] = k0
        c = conc.get(r.get('op_id')) or {}
        for fld, tag in (('input_shape', 'i'), ('output_shape', 'o')):
            sv, cv = r.get(fld), c.get(fld)
            if not sv or not cv:
                continue
            for si, (sh, cc) in enumerate(zip(sv, cv)):
                if not isinstance(sh, list) or not isinstance(cc, list) or len(sh) != len(cc):
                    continue
                for ax in range(len(sh)):
                    m[k0 + (tag, si, ax)] = (str(sh[ax]), cc[ax])
    return rows, conc, kp, m


ents, stat = {}, collections.Counter()
for ph in ('prefill', 'decode'):
    brows, bconc, bkp, bm = load(BASE, ph, SB)
    _, _, _, pm = load(PROBE, ph, SP)
    cls = ac.name_conflicts(brows, bconc)
    for v in cls.values():
        keys = []
        trusted = set()
        for (oid, f, si, ax, op, mod) in v['sites']:
            k = bkp[oid] + (f, si, ax)
            keys.append(k)
            p = pm.get(k)
            if p and ev(p[0]) == p[1]:
                trusted.add(p[0])
        if len(trusted) != 1:
            stat['판정없음' if not trusted else '판정충돌'] += 1
            continue
        to = next(iter(trusted))
        wrong = [k for k in keys if bm.get(k) and bm[k][0] != to]
        if not wrong:
            stat['이미맞음'] += 1
            continue
        # **반증 요구**: probe 가 실제로 base 이름을 깨뜨린 자리가 하나는 있어야 한다.
        # 없으면 probe 에서도 두 이름의 값이 같다는 뜻이고, 그때 이름을 바꾸는 것은 판별이
        # 아니라 probe 실행의 라벨러 잡음을 베끼는 것이다. (g_o 와 T/m_hca+1 은 T=1920 에서도
        # 둘 다 16, d_rope 와 n_h_I 는 둘 다 64 -- 이런 자리가 15개 있었다.)
        if not any(pm.get(k) and ev(bm[k][0]) is not None and ev(bm[k][0]) != pm[k][1] for k in wrong):
            stat['반증없음'] += 1
            continue
        stat['교정'] += 1
        # 대표 자리 하나에 spread:class 를 건다. 클래스 전체가 따라온다.
        k = sorted(wrong, key=str)[0]
        frm, exp = bm[k]
        ents.setdefault((k, frm, to, exp), 0)
        ents[(k, frm, to, exp)] += 1

print('클래스 통계', dict(stat))
print('대표 항목', len(ents))
print('이름쌍', collections.Counter((f, t) for (_, f, t, _) in ents).most_common(10))
json.dump([[list(map(str, k[0])), k[1], k[2], k[3], n] for k, n in ents.items()],
          open(f'{PROBE}/class_fixes.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
