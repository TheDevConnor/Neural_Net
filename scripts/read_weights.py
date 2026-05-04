import struct

def read_layer(f, layer_num):
    num_inputs, num_outputs = struct.unpack('qq', f.read(16))
    n_weights = num_inputs * num_outputs
    weights = struct.unpack(f'{n_weights}d', f.read(n_weights * 8))
    biases  = struct.unpack(f'{num_outputs}d', f.read(num_outputs * 8))
    print(f'  layer {layer_num}: {num_inputs}→{num_outputs}')
    print(f'    weights: {[round(w, 3) for w in weights]}')
    print(f'    biases:  {[round(b, 3) for b in biases]}')

gates = ['and', 'nand', 'or', 'nor', 'xor']
for gate in gates:
    print(f'\n=== {gate.upper()} ===')
    with open(f'saved_weights/{gate}.bin', 'rb') as f:
        read_layer(f, 1)
        read_layer(f, 2)

print(f'\n=== ROUNDTRIP ===')
with open('saved_weights/roundtrip.bin', 'rb') as f:
    read_layer(f, 1)
    read_layer(f, 2)