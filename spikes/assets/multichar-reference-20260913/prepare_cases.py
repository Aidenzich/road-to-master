"""Deterministic matched-semantic benchmark matrix; never submits inference."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
REFS = {r['id']:r for r in json.loads((ROOT/'reference-receipts.json').read_text())}

def build_case(cohort, count, action, repeat):
    cast = PLAN['cohorts'][cohort][:count]
    subjects = [f'Person {i+1}' for i in range(count)]
    mapping = ' '.join(f'{subjects[i]} is {REFS[r]["name"]}, the character in reference image {i+1}.' for i,r in enumerate(cast))
    order = ', '.join(subjects)
    if action == 'wave':
        action_text = 'Each person raises one open hand in a friendly greeting toward the camera, with the other hand lowered.'
    elif action == 'book':
        action_text = ('Person 1 holds one open green book with both hands at chest level and looks down at its pages.' if count == 1 else
            'Person 1 holds one open green book with both hands at chest level. Person 2 points to its right-hand page with one index finger. Everyone looks at the same book; all other hands are lowered.')
    elif count == 1:
        action_text = 'Person 1 holds one plain blue ceramic mug with both hands at chest level, looking toward the camera.'
    else:
        action_text = 'Person 1 and Person 2 face slightly toward each other, gently touching Person 1\'s right palm to Person 2\'s left palm in a held high-five at shoulder height.'
        if count >= 4:
            action_text += ' Person 3 and Person 4 do a separate held high-five, Person 3\'s right palm touching Person 4\'s left palm.'
        if count in (3,5):
            action_text += f' Person {count} watches with both hands lowered.'
        action_text += ' Each paired person keeps the unused hand lowered. Every hand clearly belongs to its own person.'
    medium = ('Clean 2D anime illustration retaining the referenced character designs.' if cohort == 'anime' else
              'Photorealistic fictional television character scene with natural facial texture.')
    scene = (f'Exactly {count} distinct people are visible, each appearing once. '
        f'The left-to-right order is {order}. They share one continuous floor in a bright simple library with pale walls. '
        'Use an eye-level waist-up composition, all heads and action-relevant hands fully visible with space between faces. '
        + action_text + ' All characters are calmly posed and fully clothed. ' + medium)
    transfer = ('Preserve each reference character\'s facial features, hairstyle, hair color and clothing. '
        'Use the pictures for character appearance only, not their original pose, handheld props, background, poster lettering, borders or special effects. '
        'Show a new clean scene, not a poster or collage. Only the requested book or mug, when specified, is held; otherwise hands are empty.')
    brief = mapping + '\n' + transfer + '\n' + scene
    h3scene = scene
    for i in range(count,0,-1):
        h3scene = h3scene.replace(f'Person {i}',f'<Subject {i}>')
    defs = '\n'.join(f'<Subject {i+1}> is {REFS[r]["name"]}, the character shown in <Picture {i+1}>.' for i,r in enumerate(cast))
    retains = '\n'.join(f'<Subject {i+1}>: fully_preserved - Preserve facial features, hairstyle, hair color and clothing from <Picture {i+1}>; replace source pose, props, background, graphics and effects with the requested scene.' for i in range(count))
    h3 = (f'subject_definitions:\n{defs}\n\nsummary:\n[reference generation] The referenced characters share one friendly library scene.\n\n'
        f'retention_analysis:\n{retains}\n\ndetailed_description:\n[Shot 1] A single continuous unbroken shot. The camera is locked. {h3scene}\n'
        '0.00-0.20833333333333334s: All subjects are already holding the described pose in the opening image and calmly maintain it for the entire shot, with only subtle natural breathing. '
        'Hands remain settled at the specified contacts or objects. The camera and lighting remain stable.\n\n'
        'overall_soundscape:\nQuiet library room tone. Nobody speaks.\n\nnon_diegetic_music:\nN/A\n')
    case_id = f'{cohort}-{count:02d}-{action}-r{repeat}'
    return dict(id=case_id,cohort=cohort,count=count,action=action,repeat=repeat,cast=cast,
                references=[REFS[r]['file'] for r in cast],reference_hashes=[REFS[r]['sha256'] for r in cast],
                seed=PLAN['seeds'][repeat-1],prompt=brief,h3_prompt=h3,
                camera_ledger={'function':'inspect character appearance and hand/object binding','framing':'waist-up group',
                 'height':'eye level','lens':'neutral perspective','path':'locked','speed':0,
                 'axis':'frontal shared floor','depth':'hands / people / plain library wall',
                 'landing':'same stable pose','why_locked':'minimum-frame still comparison, no movement confound'},
                transfer_exclusions=['pose','handheld props','background','poster graphics','special effects'],
                qwen_supported=count<=3)

if __name__ == '__main__':
    cases = []
    # First repeat covers every count/action/cohort before spending on repeats.
    for repeat in (1,2):
        for action in PLAN['actions']:
            for count in PLAN['counts']:
                for cohort in PLAN['cohorts']:
                    c = build_case(cohort,count,action,repeat)
                    d = ROOT/'cases'/c['id']
                    d.mkdir(parents=True,exist_ok=True)
                    serialized = json.dumps(c,indent=2,ensure_ascii=False)+'\n'
                    if (d/'case.json').exists():
                        assert (d/'case.json').read_text() == serialized, 'Refuse changed existing case'
                    else:
                        (d/'case.json').write_text(serialized)
                        (d/'prompt.txt').write_text(c['prompt']+'\n')
                        (d/'h3-prompt.txt').write_text(c['h3_prompt'])
                    cases.append(c['id'])
    (ROOT/'case-index.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(f'{len(cases)} matched cases; 36 native Qwen, 60 H3, 60 Codex candidate arms')
