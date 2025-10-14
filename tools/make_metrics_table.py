import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def make_table(csv_path, out_png='metrics_table.png'):
    df = pd.read_csv(csv_path)
    # pivot to have scenes as rows and views as multi-cols
    # create columns like '2v_psnr','2v_ssim','2v_lpips' etc.
    rows = []
    scenes = df['scene'].unique()
    views = sorted(df['views'].unique())
    cols = []
    for v in views:
        cols += [f'{v}v_psnr', f'{v}v_ssim', f'{v}v_lpips']

    table = pd.DataFrame(index=scenes, columns=cols)

    for _, r in df.iterrows():
        scene = r['scene']
        v = r['views']
        table.at[scene, f'{v}v_psnr'] = float(r['psnr'])
        table.at[scene, f'{v}v_ssim'] = float(r['ssim'])
        table.at[scene, f'{v}v_lpips'] = float(r['lpips'])

    # fill NaN with '-' for readability
    table = table.fillna('-')

    fig, ax = plt.subplots(figsize=(max(8, len(cols)*1.2), max(2, len(scenes)*0.6)))
    ax.axis('off')
    tbl = ax.table(cellText=table.round(3).values.tolist(), colLabels=table.columns, rowLabels=table.index, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print('Saved table to', out_png)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/make_metrics_table.py results.csv [out.png]')
        sys.exit(1)
    csv_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else 'metrics_table.png'
    make_table(csv_path, out_png)
