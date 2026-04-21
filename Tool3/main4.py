import csv

def load_best(fname, sep=','):
    best = {}
    with open(fname, newline='') as f:
        reader = csv.DictReader(f, delimiter=sep)
        for r in reader:
            idx = r['Instance'].strip()
            obj = float(r['Objective'])
            best[idx] = min(best.get(idx, 1e18), obj)
    return best

# ISA dùng tab, Tabu dùng comma — tự detect
def load_auto(fname):
    with open(fname) as f:
        header = f.readline()
    sep = '\t' if '\t' in header else ','
    return load_best(fname, sep)

isa   = load_auto("batch_results_ISA.csv")
tabu  = load_auto("tabu_results.csv")

common = set(isa) & set(tabu)

losers = [(idx, isa[idx], tabu[idx], tabu[idx] - isa[idx])
          for idx in common if tabu[idx] > isa[idx]]
losers.sort(key=lambda x: -x[3])

print(f"Tabu thua ISA: {len(losers)}/{len(common)} instances\n")
print(f"{'Instance':<12} {'ISA':>10} {'Tabu':>10} {'Delta':>10}")
print("-" * 45)
for idx, isa_v, tabu_v, delta in losers[:30]:
    print(f"T_{idx:<10} {isa_v:>10.0f} {tabu_v:>10.0f} {delta:>+10.0f}")