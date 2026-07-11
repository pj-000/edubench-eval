"""Build original-label CE datasets with teacher-confirmed low-tail resampling."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

BASE=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets/b0_original_human')
QWEN=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/qwen/p0_holistic_zero_shot/all_train.jsonl')
DEEP=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/deepseek/p0_holistic_zero_shot/secondary_route.jsonl')
OUT=Path('thesis_exp/exp17_low_score_evidence/outputs/exp30_audited_low_resampling_seed42')

def readj(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def teacher(p): return {x['sample_id']:x['annotation'] for x in readj(p) if x.get('annotation') and not x.get('schema_errors')}
def writej(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')
def dup(row,n,tag):
 x=dict(row); x['id']=f"{row['id']}::{tag}-{n}"; x['target_provenance']=tag; x['audit_resample_duplicate']=True; return x

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',type=Path,default=OUT); a=ap.parse_args()
 if (BASE/'test.jsonl').exists(): raise ValueError('test forbidden')
 train=readj(BASE/'train.jsonl'); dev=readj(BASE/'dev.jsonl'); q=teacher(QWEN); d=teacher(DEEP)
 low=[r for r in train if int(r['label_5'])<=2]; reliable=[r for r in low if int(q[r['record_id']]['score'])==int(r['label_5']) or int(d[r['record_id']]['score'])==int(r['label_5'])]
 counts=Counter(int(r['label_5']) for r in reliable); random=[]
 for label,n in sorted(counts.items()):
  pool=[r for r in low if int(r['label_5'])==label]
  pool.sort(key=lambda r:hashlib.sha256(('exp30-random|'+r['record_id']).encode()).hexdigest()); random.extend(pool[:n])
 variants={'d1_teacher_confirmed_low':[dict(r) for r in train], 'd2_random_low_control':[dict(r) for r in train]}
 for name,chosen in [('d1_teacher_confirmed_low',reliable),('d2_random_low_control',random)]:
  for row in chosen:
   for n in range(3): variants[name].append(dup(row,n,name))
 root=a.out_dir/'private/datasets'; summary=[]
 for name,rows in variants.items():
  writej(root/name/'train.jsonl',rows); writej(root/name/'dev.jsonl',dev)
  c=Counter(int(r['label_5']) for r in rows); summary.append({'variant':name,'train_rows':len(rows),'duplicates':len(rows)-len(train),**{f'label_{i}':c[i] for i in range(1,6)}})
 if len(reliable)!=38 or any(len(x)!=2768 for x in variants.values()): raise ValueError('locked resampling count mismatch')
 t=a.out_dir/'tables/exp30_dataset_summary.csv'; t.parent.mkdir(parents=True,exist_ok=True)
 with t.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n'); w.writeheader(); w.writerows(summary)
 dec={'status':'READY_FOR_SEED42_DEV_SCOUT','teacher_confirmed_low_rows':len(reliable),'label_counts':dict(counts),'duplicates_per_selected_row':3,'loss':'ordinary_cross_entropy','labels_changed':0,'test_read':False}
 p=a.out_dir/'decision/exp30_dataset_decision.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(dec,indent=2)+'\n')
 print(json.dumps(dec,sort_keys=True))
if __name__=='__main__': main()
