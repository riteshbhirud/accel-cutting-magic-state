import sys
import re
import subprocess

sys.path.insert(0, '/Users/ritesh/Downloads/prx/accel-cutting-magic-state')
sys.path.insert(0, '/Users/ritesh/Downloads/prx/tsim/src')

# Step 1: Build T-gate circuit string
D3_CLIFFORD = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/for_perfectionist_decoding/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
with open(D3_CLIFFORD) as f:
    content = f.read()
t_content = re.sub(r'\bS_DAG\b', 'T_DAG', content)
t_content = re.sub(r'\bS\b', 'T', t_content)

# Step 2: Look at conversion calls
result = subprocess.run(
    ['grep', '-n', 'get_graph\\|prepare_graph\\|tsim.Circuit\\|from_stim\\|stim_to',
     '/Users/ritesh/Downloads/prx/accel-cutting-magic-state/tsim_cutting.py'],
    capture_output=True, text=True
)
print("Conversion calls in tsim_cutting.py:")
print(result.stdout)

result2 = subprocess.run(
    ['grep', '-rn', 'get_graph\\|prepare_graph\\|from_stim\\|stim_to_zx',
     '/Users/ritesh/Downloads/prx/tsim/src/'],
    capture_output=True, text=True
)
print("\nConversion calls in tsim/src:")
print(result2.stdout[:3000])
