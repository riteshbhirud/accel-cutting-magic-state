import stim
import numpy as np

D3 = "/Users/ritesh/Downloads/prx/gidney-circuits/circuits/for_perfectionist_decoding/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"

circuit = stim.Circuit(open(D3).read())

# Method 1: stim direct sampler (ground truth)
N = 100000

det_sampler = circuit.compile_detector_sampler()
detection_events, observable_flips = det_sampler.sample(N, separate_observables=True)

# Postselection: discard shots where ANY detector fired
no_detection = ~np.any(detection_events, axis=1)
n_postselected_stim = no_detection.sum()
n_logical_errors_stim = observable_flips[no_detection, 0].sum()

psr_stim = n_postselected_stim / N
ler_stim = n_logical_errors_stim / n_postselected_stim if n_postselected_stim > 0 else 0

print(f"STIM DIRECT (ground truth, N={N}):")
print(f"  PSR = {psr_stim:.4f}  (Wan & Zhong Table 3 target: 0.651)")
print(f"  LER = {ler_stim:.2e}  (Wan & Zhong Table 3 target: ~1.0e-6)")
print(f"  Postselected shots: {n_postselected_stim}")
print(f"  Logical errors: {n_logical_errors_stim}")
