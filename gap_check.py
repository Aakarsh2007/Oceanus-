with open('dashboard/index.html', encoding='utf-8') as f:
    c = f.read()

missing = []
done = []

checks = [
    # Already done
    ('dominant', 'Bio dominant 2x card'),
    ('font-size:36px', 'Bio 36px hero'),
    ('secondary', 'Secondary cards dimmed'),
    ('story-panel', 'Story panel'),
    ('updateStoryPanel', 'Story JS'),
    ('policy-panel', 'Policy panel'),
    ('demo-btn', 'Demo button'),
    ('transport-btn', 'Video player controls'),
    ('replay-markers', 'Key moment markers'),
    ('replay-narrative', 'Replay narrative'),
    ('humanReadableEvent', 'Human readable events'),
    ('baseline-box', 'Compare colored borders'),
    ('Ecosystem Collapse', 'Emotional compare labels'),
    ('height:160px', 'Taller charts'),
    ('borderDash', 'Dashed baseline line'),
    ('story-fade', 'Story animation'),
    # Still missing
    ('reasoning', 'Agent reasoning text'),
    ('last-action', 'Agent last action display'),
    ('coordination', 'Coordination indicator'),
    ('trail', 'ASV movement trails'),
    ('broadcast-line', 'Broadcast coordination lines'),
    ('clean-pulse', 'Clean pulse animation'),
    ('chaos-event-card', 'Chaos event cards'),
    ('storm-overlay', 'Storm visual effect'),
    ('policy-timeline', 'Policy timeline steps'),
    ('cubic-bezier', 'Smooth easing motion'),
    ('0B2A2F', 'Ocean radial gradient bg'),
    ('chart-annotation', 'Chart annotations'),
    ('cause-effect', 'Causality annotations'),
]

for key, label in checks:
    if key in c:
        done.append(label)
    else:
        missing.append((key, label))

print(f"DONE ({len(done)}):")
for l in done: print(f"  [x] {l}")
print(f"\nMISSING ({len(missing)}) — need to implement:")
for k, l in missing: print(f"  [ ] {l}  ({k})")
