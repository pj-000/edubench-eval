"""Build original-label datasets for teacher-disputed low-tail resampling."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
BASE=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets/b0_original_human')
Q=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/qwen/p0_holistic_zero_shot/all_train.jsonl')
D=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/deepseek/p0_holistic_zero_shot/secondary_route.jsonl')
OUT=Path('thesis_exp/exp17_low_score_evidence/outputs/exp31_disputed_low_resampling_seed42')
def rj(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def teach(p): return {x['sample_id']:x['annotation'] for x in rj(p) if x.get('annotation') and not x.get('schema_errors')}
def wj(p,rows): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows))
def main():
 if (BASE/'test.jsonl').exists(): raise ValueError('test forbidden')
 train=rj(BASE/'train.jsonl'); dev=rj(BASE/'dev.jsonl'); q=teach(Q); d=teach(D); low=[x for x in train if int(x['label_5'])<=2]
 hard=[x for x in low if int(q[x['record_id']]['score'])!=int(x['label_5']) and int(d[x['record_id']]['score'])!=int(x['label_5'])]
 counts=Counter(int(x['label_5']) for x in hard); random=[]
 for y,n in counts.items():
  pool=[x for x in low if int(x['label_5'])==y]; pool.sort(key=lambda x:hashlib.sha256(('exp31|'+x['record_id']).encode()).hexdigest()); random+=pool[:n]
 vs={'e1_teacher_disputed_low':hard,'e2_random_low_control':random}; root=OUT/'private/datasets'; rows=[]
 for name,chosen in vs.items():
  data=[dict(x) for x in train]
  for x in chosen:
   for i in range(3): z=dict(x); z['id']=f"{x['id']}::{name}-{i}"; z['target_provenance']=name; data.append(z)
  wj(root/name/'train.jsonl',data); wj(root/name/'dev.jsonl',dev); c=Counter(int(x['label_5']) for x in data); rows.append({'variant':name,'train_rows':len(data),'duplicates':len(data)-2654,**{f'label_{i}':c[i] for i in range(1,6)}})
 if len(hard)!=38 or any(r['train_rows']!=2768 for r in rows): raise ValueError('count mismatch')
 p=OUT/'tables/exp31_dataset_summary.csv'; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
 dec={'status':'READY_FOR_SEED42_DEV_SCOUT','teacher_disputed_low_rows':38,'label_counts':dict(counts),'labels_changed':0,'loss':'ordinary_cross_entropy','test_read':False}; p=OUT/'decision/exp31_dataset_decision.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(dec,indent=2)+'\n'); print(json.dumps(dec))
if __name__=='__main__': main()
