fw = open('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/future/T2306.CFE_20230409_20230419_new.csv', 'w')

last_t = None
n = 0
with open('/Users/tuxun/work/jnwork/建模大赛/HFtrading/data/future/T2306.CFE_20230409_20230419.csv') as f:
    for line in f:
        if n == 0:
            fw.write(line)
            n = 1
            continue
        cols = line.strip().split(',')
        t = cols[0]
        if t == last_t:
            t1 = t + '.5'
            outline = t1 + ',' + ','.join(cols[1:]) + '\n'
            last_t = t
            fw.write(outline)
        else:
            t1 = t + '.0'
            outline = t1 + ',' + ','.join(cols[1:]) + '\n'
            last_t = t
            fw.write(outline)