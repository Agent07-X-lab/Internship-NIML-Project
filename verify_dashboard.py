import sys

path = '3_processed_outputs/Predictive_Maintenance_Dashboard.html'
with open(path, encoding='utf-8') as f:
    content = f.read()

checks = [
    ('isolated_nodes in data', 'isolated_nodes' in content),
    ('connected_nodes in data', 'connected_nodes' in content),
    ('buildGraphData function', 'function buildGraphData' in content),
    ('EDGE_THRESHOLD const', 'EDGE_THRESHOLD' in content),
    ('Self-loop guard', 'Self-loop' in content),
    ('No duplicate renderOverviewGraph', content.count('function renderOverviewGraph') == 1),
    ('No duplicate renderEvolutionGraph', content.count('function renderEvolutionGraph') == 1),
]

all_ok = True
for desc, result in checks:
    status = 'OK' if result else 'FAIL'
    if not result:
        all_ok = False
    print(f'  [{status}] {desc}')

if all_ok:
    print('\nAll checks passed.')
else:
    print('\nSome checks failed.')
    sys.exit(1)
