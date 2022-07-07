import subprocess
import os
import dm_local_constants as dm_cst


fin = dm_cst.fin_ZA
fout = dm_cst.fout_ZA
gdal_path = dm_cst.gdal_path
gdal_calc_path = os.path.join(gdal_path, 'gdal_calc.py')
gdal_calc_str = 'python {0} -A {1} --outfile={2} --calc={3} --overwrite'
calc_expr = '"A * numpy.isin(A, [1712, 1713, 2264, 2265])"'

gdal_calc_process = gdal_calc_str.format(gdal_calc_path, fin,
        fout, calc_expr)
        # Call process.
os.system(gdal_calc_process)



run_cmd = [
    'gdal_calc',
    '-A',
    fin,
    '--format=ENVI',
    '--outfile=' + fout,
    '--calc="A * numpy.isin(A, [1712, 1713, 2264, 2265])"'
]

p = subprocess.run(run_cmd, shell=False, input='\n', capture_output=True, text=True)
if p.returncode != 0:
    print(p)