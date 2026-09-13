"""Export reviewed evidence to road-to-master. Never commits or pushes by itself."""
from pathlib import Path
import csv, hashlib, json, shutil
from statistics import median
from datetime import datetime, timezone
from quality_summary import summarize

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/aiden/Projects/road-to-master')
SLUG='multichar-reference-20260913'
DEST=REPO/'spikes/assets'/SLUG
PLAN=json.loads((ROOT/'plan.json').read_text())
CASES=json.loads((ROOT/'case-index.json').read_text())
MODEL_NAMES={'codex':'Codex built-in','qwen':'Qwen Edit 2511','h3':'H3 Ref2VA 5 frames'}
METRICS=['exact_count','appearance_preserved','reference_binding','action_obedience','hands_and_contacts']

def copy_file(relative):
    source=ROOT/relative
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)

for relative in ['plan.json','references.json','reference-receipts.json','case-index.json','controls.json','prepare_cases.py','gpu_runner.py','fetch_references.py','record_review.py','codex_receipt.py','publish_report.py']:
    copy_file(relative)
for relative in ['singular-control-plan.json','summarize_timings.py','timings.json','timings.csv',
                 'position-control-index.json','prepare_position_controls.py',
                 'audit_evidence.py','evidence-audit.json','quality_summary.py','test_quality_summary.py',
                 'audit_remote_cleanup.py','remote-cleanup-audit.json',
                 'audit_workflows.py','workflow-audit.json']:
    if (ROOT/relative).exists():
        copy_file(relative)
for reference in json.loads((ROOT/'reference-receipts.json').read_text()):
    if reference['state']=='downloaded':
        copy_file(reference['file'])
rows=[]
for case_id in CASES:
    for filename in ('case.json','prompt.txt','h3-prompt.txt'):
        copy_file(f'cases/{case_id}/{filename}')
    c=json.loads((ROOT/'cases'/case_id/'case.json').read_text())
    for model in MODEL_NAMES:
        directory=ROOT/'runs'/model/case_id
        result_file=directory/'result.json'
        r=json.loads(result_file.read_text()) if result_file.exists() else {}
        state=r.get('state','pending')
        if not r and model=='qwen' and not c['qwen_supported']:
            state='unsupported'
        row=dict(case_id=case_id,model=model,cohort=c['cohort'],count=c['count'],action=c['action'],repeat=c['repeat'],
            state=state,submitted=r.get('submitted',False),catalog_state=r.get('catalog_state'),failure_category=r.get('failure_category'),
            seconds=r.get('elapsed_seconds',r.get('tool_elapsed_seconds')),reviewed=bool(r.get('review')))
        for metric in METRICS:
            row[metric]=(r.get('review') or {}).get(metric)
        row['notes']=(r.get('review') or {}).get('notes',r.get('notes',''))
        rows.append(row)
        if directory.exists():
            # Explicit publication allowlist; never publish database blobs/journals or private env.
            for filename in ('run.json','result.json','events.json','history.json','cleanup.json','ffprobe.json',
                             'output.png','output.mp4','frame-01.png','frame-02.png','frame-03.png','frame-04.png','frame-05.png'):
                if (directory/filename).exists():
                    copy_file(f'runs/{model}/{case_id}/{filename}')
DEST.mkdir(parents=True,exist_ok=True)
with (DEST/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows)
(DEST/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
quality = summarize(rows)
(DEST/'quality-summary.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n')
now=datetime.now(timezone.utc).isoformat()
asset=f'assets/{SLUG}'
lines=[
'# 六小時多角色參考圖實驗：Codex／Qwen Edit／H3', '',
f'資料更新：{now}。**持續實驗中，非最終結論。**', '',
'## 問題與方法', '',
'比較同一組參考角色在 1–5 人、不同互動動作下的外觀保留、數量、對應與手部表現。六小時窗口：台灣時間 2026-09-13 23:41:32 至 2026-09-14 05:41:32；到期後不新增提交，已提交任務完成、檢查與清理另計。', '',
'兩類角色 × 五種人數 × 三類動作 × 兩次重複，預先登記 60 個場景；完整窮舉所有角色子集／排列將遠超時窗，因此先做分層覆盖，再依剩餘時間增加單變因對照。未執行、流程不支援、執行失敗與未人工檢查皆分開列出。', '',
'三類動作：各自揮手；共同看同一本綠色書（單人為自己看書）；兩兩擊掌（單人為雙手持藍色杯）。比較靜態完成姿勢，並非完整動作序列。H3 要求首幀即持有該姿勢並保持 5 幀；固定眼平構圖。', '',
'| 模型 | 實驗配置 | 限制 |', '|---|---|---|',
'| Codex 內建生圖 | 同一語意 prompt + 3:2 構圖要求 | 工具未提供可核實 seed、steps、CFG、GPU 與底層模型版本；實際尺寸逐張記錄 |',
'| Qwen Image Edit 2511 FP8 mixed | 768×512、40 steps、CFG4、Euler/simple、denoise1、shift3.1、CFGNorm1、batch1 | 現有 adapter／TextEncodeQwenImageEditPlus 接受至多3張獨立參考圖；4–5張原生條件不支援，不代表模型架構永遠不可能 |',
'| H3 Ref2VA INT8 ConvRot | 768×512、20 steps、Euler/linear_quadratic、5幀、24fps、batch1、無Turbo LoRA | 專用六欄 prompt；5幀低於常規片長，視為取圖實驗。保留全部5幀及MP4，主比較固定第1幀，不挑最佳幀 |', '',
'本地模型使用 RTX 5090，共享 ComfyUI 的 vision-flow 裝置策略；H3 low-VRAM／reserve6GiB，Qwen 使用既有 image 策略。工作流快照含實際編碼器裝置選項與硬體資訊。不重啟服務、不全域中斷 GPU。六個場景為一小模型批段並交替模型順序，以降低頻繁換模；時間漂移與cache差異仍是限制。', '',
'## 來源與素材限制', '',
'- [芙莉蓮官方角色頁](https://frieren-anime.jp/character/chara_group1/1-1/)與[菲倫](https://frieren-anime.jp/character/chara_group1/1-5/)；[2026官方消息](https://frieren-anime.jp/news/)。',
'- [SPY×FAMILY 官方頁](https://spy-family.net/tvseries/)：Loid、Yor、Anya。',
'- [Netflix 官方 Wednesday Season 2 角色頁](https://www.netflix.com/tudum/articles/wednesday-season-2-character-cast-guide)：Wednesday、Enid、Bianca、Tyler、Dort。',
'- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)列40 steps／CFG4範例；本地量化版本不等於官方BF16配置。', '',
f'原始下載URL、尺寸與SHA-256見 [reference-receipts.json]({asset}/reference-receipts.json)。官方圖版權屬各權利人，僅作本報告的具名比較與來源證據；不是本專案原創或可再授權素材。生成結果為虛構場景。', '',
'動漫素材含透明背景，實際RGB／alpha處理由各adapter不同流程處理，並非完全相同的編碼輸入；真人素材是含大型字母、品牌、特殊效果的官方海報，不是乾淨肖像。原圖姿勢與道具均明確不要求保留，且未用任一比較模型先重畫參考圖。這些素材品質／前處理差異須列為解讀限制；不能將結果只歸因於模型本身。Tyler參考服裝限制手部活動，是額外的姿勢轉移難例。', '',
'## 進度與分母', '',
'| 模型 | 預登記格數 | 已提交 | 推論／取圖成功 | 執行失敗 | 不支援 | 已目視評估 |', '|---|---:|---:|---:|---:|---:|---:|']
for model,name in MODEL_NAMES.items():
    sub=[r for r in rows if r['model']==model]
    lines.append(f'| {name} | {len(sub)} | {sum(bool(r["submitted"]) for r in sub)} | {sum(r["state"]=="succeeded" for r in sub)} | {sum(r["state"]=="failed" for r in sub)} | {sum(r["state"]=="unsupported" for r in sub)} | {sum(r["reviewed"] for r in sub)} |')
if (ROOT/'evidence-audit.json').exists():
    audit=json.loads((ROOT/'evidence-audit.json').read_text())
    lines += ['', '### 資料完整性快照', '',
              f'檢查 {len(audit["checks"])} 項，記錄 {len(audit["files"])} 個成果檔案雜湊；{len(audit["failed"])} 項未通過。此快照不代表實驗完成，也不等於重新連線驗證遠端刪除。', '',
              f'[完整檢查與失敗清單]({asset}/evidence-audit.json) · [檢查程式]({asset}/audit_evidence.py)', '',
              '早期8筆Codex多角色揮手請求只保存參考圖路徑，缺少提交當下的reference_hashes；現在原圖與下載紀錄雜湊一致，但不能以事後計算補造當時的傳輸證據。這8筆保留成果與缺漏標記，不宣稱完全可追溯。']
if (ROOT/'workflow-audit.json').exists():
    workflow_audit=json.loads((ROOT/'workflow-audit.json').read_text())
    lines += ['', f'實際GPU工作流核查：{len(workflow_audit["checks"])} 項，{len(workflow_audit["failed"])} 項未通過；包括provider已接收圖與prepared圖一致、prompt、參考圖連線／順序／上傳雜湊、seed與採樣設定。這證明配置連線，不證明模型一定保留角色。', '',
              f'[實際工作流核查]({asset}/workflow-audit.json) · [核查程式]({asset}/audit_workflows.py)']
if (ROOT/'remote-cleanup-audit.json').exists():
    remote_audit=json.loads((ROOT/'remote-cleanup-audit.json').read_text())
    lines += ['', f'遠端清理獨立快照：{len(remote_audit["files"])} 個已完成任務檔案的本機封存bytes／SHA-256通過核對，逐一SSH檢查後仍存在的遠端檔案為 {len(remote_audit["present"])}。僅包含終止且已有清理憑證的自有輸入／輸出；不包含正在生成的任務，也不代表整台5090為空。', '',
              f'[逐檔封存與遠端不存在證據]({asset}/remote-cleanup-audit.json) · [唯讀檢查程式]({asset}/audit_remote_cleanup.py)']
lines += ['', '評分：0明確失敗、1部分符合或不確定、2明確符合；null未審查／不適用。人工目視評分不是生物辨識身份驗證，也不是盲測或多評審共識。尚未有足夠重複樣本前，不宣稱統計顯著或模型優劣排名。', '',
f'[完整CSV]({asset}/results.csv) · [JSON]({asset}/results.json) · [預登記計畫]({asset}/plan.json)', '']
lines += ['## 分組評分（明確符合數／已評估數）', '',
          '只把評分2計為明確符合；1是部分符合或不確定，0是明確失敗。沒有圖、未審查與不支援不當作視覺0分，也不藏入已評估分母。以下原生流程表不包含額外對照。', '',
          f'[人數／動作／重複批次的完整0/1/2及缺漏計數]({asset}/quality-summary.json)', '']
for scope,title in [('all_available_baseline','目前可用的全部原生流程成果'),('three_model_matched_complete','三模型均成功且已評估的相同場景')]:
    lines += [f'### {title}', '']
    if scope == 'three_model_matched_complete':
        lines += [f'目前交集為 {len(quality["matched_case_ids"])} 個場景。此表以成功且完成評估為條件，會排除失敗及不支援條件，存在完整案例選擇偏差；不是完整成功率或公平模型排名。尺寸、前處理、量化與seed可控性等差異仍存在。', '']
    lines += ['| 人數 | 模型 | 格數 | 數量 | 外觀 | 對應 | 動作 | 手部 |', '|---:|---|---:|---:|---:|---:|---:|---:|']
    for g in quality['groups']:
        if g['scope'] != scope or g['dimension'] != 'count':
            continue
        scores = [f'{g["metrics"][m]["clear_pass"]}/{g["metrics"][m]["reviewed"]}' if g['metrics'][m]['reviewed'] else '—（0已評估）' for m in METRICS]
        lines.append(f'| {g["value"]} | {MODEL_NAMES[g["model"]]} | {g["arms"]} | ' + ' | '.join(scores) + ' |')
    lines += ['']
lines += ['## 個別結果', '', '| 場景 | Codex | Qwen | H3 |', '|---|---|---|---|']
for cid in CASES:
    cells=[]
    for model in MODEL_NAMES:
        r=next(r for r in rows if r['case_id']==cid and r['model']==model)
        rel=f'{asset}/runs/{model}/{cid}'
        cells.append(f'[{r["state"]}]({rel}/result.json)' if (DEST/'runs'/model/cid/'result.json').exists() else r['state'])
    lines.append('| '+cid+' | '+' | '.join(cells)+' |')
lines += ['', '## 圖片對照（包含失败成像，不做優勝挑選）', '']
for cid in CASES:
    if not any((DEST/'runs'/m/cid/'output.png').exists() for m in MODEL_NAMES):
        continue
    lines += [f'### {cid}', '', f'[共同要求]({asset}/cases/{cid}/prompt.txt) · [H3實際prompt]({asset}/cases/{cid}/h3-prompt.txt)', '',
              '| Codex | Qwen | H3首幀 |','|---|---|---|']
    cells=[]
    for model in MODEL_NAMES:
        f=DEST/'runs'/model/cid/'output.png'
        cells.append(f'![{model}]({asset}/runs/{model}/{cid}/output.png)' if f.exists() else '未產出／待執行')
    lines += ['| '+' | '.join(cells)+' |','']
lines += ['## 單變因對照：第一張參考圖的裁切', '',
          '此分支不更動產品。只將 Qwen 正／負文字編碼節點的 image1 改接完整第一張載入圖，輸出 latent 的尺寸／裁切保持不變。相同場景的 prompt、參考檔案與順序、seed、steps、CFG 不變。這些是額外對照，不計入上述原生流程的分母。', '',
          f'[預登記對照]({asset}/controls.json)', '',
          '| 場景 | 原流程 | 完整第一參考圖 | 觀察 |', '|---|---|---|---|']
for result_file in sorted((ROOT/'runs/qwen-fullref').glob('*/result.json')):
    cid=result_file.parent.name
    result=json.loads(result_file.read_text())
    for filename in ('run.json','result.json','events.json','history.json','cleanup.json','output.png'):
        if (result_file.parent/filename).exists():
            copy_file(f'runs/qwen-fullref/{cid}/{filename}')
    notes=(result.get('review') or {}).get('notes','尚未目視評估').replace('|','/')
    baseline=f'![baseline]({asset}/runs/qwen/{cid}/output.png)' if (DEST/'runs/qwen'/cid/'output.png').exists() else '尚未產出'
    control=f'![fullref]({asset}/runs/qwen-fullref/{cid}/output.png)' if (result_file.parent/'output.png').exists() else result['state']
    lines.append(f'| {cid} | {baseline} | {control} | {notes} |')
position_index=ROOT/'position-control-index.json'
if position_index.exists():
    position_rows=[]
    lines += ['', '## 額外對照：只反轉畫面左右排列', '',
              '參考圖順序、角色編號、持書／指向／擊掌等動作角色與生成參數均維持原值，只改左右排列一句。目標是測試模型是否能脫離參考圖輸入順序排人；並非變更參考圖的上傳順序。Codex 沒有可控制seed，因此單張差異仍有隨機因素。這些額外對照不計入固定矩陣分母。', '',
              f'[預登記與固定條件]({asset}/position-control-index.json)', '']
    for control in json.loads(position_index.read_text())['controls']:
        cid=control['case_id']
        baseline_id=control['baseline_case_id']
        for filename in ('case.json','prompt.txt','h3-prompt.txt'):
            copy_file(f'cases/{cid}/{filename}')
        lines += [f'### {cid}', '',
                  f'[對照prompt]({asset}/cases/{cid}/prompt.txt) · [原始條件]({asset}/cases/{baseline_id}/prompt.txt)', '',
                  '| 模型 | 原始排列 | 反轉排列 | 觀察 |', '|---|---|---|---|']
        for model,name in MODEL_NAMES.items():
            directory=ROOT/'runs'/model/cid
            result_path=directory/'result.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else {}
            state=result.get('state','not_executed')
            if not result and model=='qwen' and not control['qwen_supported']:
                state='unsupported'
            for filename in ('run.json','result.json','events.json','history.json','cleanup.json','ffprobe.json',
                             'output.png','output.mp4','frame-01.png','frame-02.png','frame-03.png','frame-04.png','frame-05.png'):
                if (directory/filename).exists():
                    copy_file(f'runs/{model}/{cid}/{filename}')
            row=dict(case_id=cid,baseline_case_id=baseline_id,model=model,state=state,
                     submitted=result.get('submitted',False),review=result.get('review'))
            position_rows.append(row)
            baseline=f'![baseline]({asset}/runs/{model}/{baseline_id}/output.png)' if (ROOT/'runs'/model/baseline_id/'output.png').exists() else '尚未產出'
            output=f'![reverse]({asset}/runs/{model}/{cid}/output.png)' if (directory/'output.png').exists() else state
            notes=(result.get('review') or {}).get('notes',result.get('notes','')).replace('|','/')
            lines.append(f'| {name} | {baseline} | {output} | {notes} |')
        lines += ['']
    (DEST/'position-control-results.json').write_text(json.dumps(position_rows,indent=2,ensure_ascii=False)+'\n')
lines += ['', '## 耗時與硬體紀錄', '']
if (ROOT/'timings.json').exists():
    timing_rows=json.loads((ROOT/'timings.json').read_text())
    case_counts={cid:json.loads((ROOT/'cases'/cid/'case.json').read_text())['count'] for cid in CASES}
    lines += ['以下只含固定矩陣成功項目，不混入額外對照。每格為中位秒數 [最小–最大]；括號n是該欄有值的樣本數。缺失階段不填0。參考圖數與角色組合一起改變，因此不能把差值全歸因於圖數。', '',
              '| 模型 | 人數／參考圖 | GPU工作流執行 | 採樣節點觀測（可能含載模） | 首步至末步 | runner總耗時 |',
              '|---|---:|---|---|---|---|']
    def timing_cell(subset,key):
        values=[r[key] for r in subset if isinstance(r.get(key),(int,float))]
        return f'{median(values):.2f} [{min(values):.2f}–{max(values):.2f}] (n={len(values)})' if values else '未知 (n=0)'
    for model in ['qwen','h3']:
        for count in range(1,6):
            subset=[r for r in timing_rows if r['model']==model and r['state']=='succeeded' and case_counts.get(r['case_id'])==count]
            values=[timing_cell(subset,key) for key in ['provider_execution_seconds','sampling_node_seconds','first_to_last_step_seconds','runner_seconds']]
            lines.append(f'| {MODEL_NAMES[model]} | {count} | '+' | '.join(values)+' |')
    lines += ['', 'Codex另表：只有工具牆鐘時間，沒有相同GPU／階段／解析度控制，不能由下表得出等算力速度比。', '',
              '| 人數／參考圖 | 內建工具牆鐘秒數 |','|---:|---|']
    for count in range(1,6):
        subset=[r for r in timing_rows if r['model']=='codex' and r['state']=='succeeded' and case_counts.get(r['case_id'])==count]
        lines.append(f'| {count} | {timing_cell(subset,"builtin_tool_wall_seconds")} |')
    lines += ['']
lines += [
f'[逐筆階段耗時 CSV]({asset}/timings.csv) · [JSON]({asset}/timings.json) · [推導腳本]({asset}/summarize_timings.py)', '',
'provider_execution_seconds 取同一 prompt_id 的 execution_start 至 execution_success；provider_queue_seconds 取服務收件 create_time 至 execution_start。sampling_node_seconds 是採樣節點觀測區間，可能包含載模，並非純 CUDA kernel 時間；first_to_last_step_seconds 不含第一步之前的準備。collection_to_saved_seconds 只在兩事件都存在時提供。Codex 僅有內建工具牆鐘時間，沒有相同階段或硬體資訊，不作等算力速度排名。VRAM 是提交前快照，不是峰值；未知值保留 null。', '',
'## 重現與失敗歸類', '',
'內建生圖拒絕、基礎設施錯誤、成功成像但品質不符是不同結果。`failure_category=provider_output_moderation_blocked` 表示服務輸出階段拒絕，沒有可評分圖片；不得算成人物一致性零分，也不自動改寫提示詞繞過或切換API。完整錯誤代碼與request ID保留在該筆result.json。', '',
f'場景與角色對應可由 [prepare_cases.py]({asset}/prepare_cases.py) 重建；[gpu_runner.py]({asset}/gpu_runner.py) 使用現有 Veritas adapter、PostgreSQL 的自有 schema 與本機清理 journal，需自行提供本地服務配置（此PR不含env或密鑰）。腳本含作者環境路徑，移植時須調整，不能當作通用一鍵執行套件。Codex 使用內建 image_gen 逐張呼叫，實際prompt与來源順序保存在各run.json，不宣稱可由seed重現。', '',
'H3 5幀取圖保存原始MP4；若音軌0.20秒短於5/24秒，現有一般影片collector會拒絕影音等長檢查。此時分別記錄provider成功與catalog失敗，從已驗證的本機journal影片取圖；不重試、不補幀、不放寬產品校驗。只有實際解出5幀才記為取圖成功。所有已完成實验的遠端輸入／輸出需有hash比對與清理收據，不能用刪整個資料夾代替。', '']
observations=['## 目前觀察（非最終結論）', '',
              '這是特定量化模型、參考素材與前處理工作流的比較，不是模型排行榜。以下只統計已成功成像且已評估的原生流程；執行失敗、不支援與未執行仍保留在完整分母表。', '',
              '| 已評估條件 | Codex | Qwen | H3 |', '|---|---:|---:|---:|']
for count,metric,label in [(1,'exact_count','單人：恰好一人的明確符合數'),
                           (3,'reference_binding','三人：角色對應的明確符合數'),
                           (3,'action_obedience','三人：全部動作要求的明確符合數')]:
    cells=[]
    for model in MODEL_NAMES:
        group=next(g for g in quality['groups'] if g['scope']=='all_available_baseline' and g['dimension']=='count' and g['value']==count and g['model']==model)
        metric_counts=group['metrics'][metric]
        cells.append(f'{metric_counts["clear_pass"]}/{metric_counts["reviewed"]}' if metric_counts['reviewed'] else '未評估')
    observations.append('| '+label+' | '+' | '.join(cells)+' |')
observations += ['',
    '- 人數與角色大致可辨，不代表細節、指定左右手或旁觀者動作正確；手部解剖正常也可能配錯角色。逐圖註記保留這些差異。',
    '- 單人重複成多人是目前Qwen／H3原生流程反覆出現的失敗；不能據此斷言模型無法生成單人。原始提示詞含泛用複數表述，且Qwen第一張圖經中心裁切，皆是待分離的混雜因素。',
    '- 一筆Qwen完整第一張參考圖的對照改善了人物細節，但只有單一案例，尚不能證明普遍改善或公平速度優勢。',
    '- Qwen現有adapter最多3張獨立參考圖；4–5張標為不支援，沒有暗中減少圖片或改成拼貼。Codex沒有可控制seed，不能把兩張圖的差異當成嚴格同噪聲因果實驗。', '',
    '[跳至分組評分](#分組評分明確符合數已評估數) · [跳至完整圖片對照](#圖片對照包含失败成像不做優勝挑選) · [耗時與硬體](#耗時與硬體紀錄)', '']
lines[4:4]=observations
(REPO/'spikes/multichar-reference-benchmark-20260913.md').write_text('\n'.join(lines)+'\n')
# Same complete report locally, with portable links to this experiment's files.
for filename in ['results.csv','results.json','quality-summary.json','position-control-results.json']:
    if (DEST/filename).exists():
        shutil.copyfile(DEST/filename,ROOT/filename)
(ROOT/'README.md').write_text(('\n'.join(lines)+'\n').replace(f']({asset}/',']('))
print(f'Exported {len(rows)} planned arms; {sum(r["state"]=="succeeded" for r in rows)} successful results')
