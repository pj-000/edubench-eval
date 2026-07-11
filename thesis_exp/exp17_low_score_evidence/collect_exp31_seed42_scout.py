"""Collect Exp31 seed42 scout."""
import csv,json
from pathlib import Path
O=Path('thesis_exp/exp17_low_score_evidence/outputs/exp31_disputed_low_resampling_seed42'); B=Path('thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/runs/b0_original_human/seed_42/metrics.json'); V=('e1_teacher_disputed_low','e2_random_low_control')
def main():
 rows=[{'variant':'b0_original_human',**json.loads(B.read_text())[0]}]+[{'variant':v,**json.loads((O/f'runs/{v}/seed_42/metrics.json').read_text())[0]} for v in V]; p=O/'tables/exp31_seed42_dev_metrics.csv'; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
 d={x['variant']:x for x in rows}; b=d['b0_original_human']; m=d[V[0]]; c=d[V[1]]; checks={'mae_guard':m['MAE_label']<=b['MAE_label']+.01,'exact_guard':m['Exact Match']>=b['Exact Match']-.01,'kendall_guard':m['Kendall tau']>=b['Kendall tau']-.01,'low_to_high_improves':m['low_to_high_rate']<b['low_to_high_rate'],'label2_improves':m['Acc@2']>b['Acc@2'],'beats_random_mae':m['MAE_label']<c['MAE_label'],'beats_random_low_risk':m['low_to_high_rate']<c['low_to_high_rate']}; dec={'status':'READY_FOR_SEEDS_43_44' if all(checks.values()) else 'SEED42_SCOUT_NOT_SUPPORTED','checks':checks,'test_read':False}; p=O/'decision/exp31_seed42_scout_decision.json'; p.write_text(json.dumps(dec,indent=2)+'\n'); print(json.dumps(dec))
if __name__=='__main__': main()
