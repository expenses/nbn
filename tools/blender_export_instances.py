import bpy
import json
import numpy as np
import os

obj = bpy.context.active_object

depsgraph = bpy.context.evaluated_depsgraph_get()

m = {}
total = 0

last = None

for inst in depsgraph.object_instances:
    if not inst.is_instance:
        continue

    name = inst.object.name
    
    if inst.parent.name != last:
        print(inst.parent.name)
        last = inst.parent.name

    if not name in m:
        m[name] = []
    
    # Copy is required otherwise the same matrix just gets duplicated
    m[name].append(inst.matrix_world.copy())
    total += 1

np_data = {
    name: np.array(matrices, dtype=np.float32).ravel()
    for name, matrices in m.items()
}

filepath = bpy.path.abspath(os.path.join("//", "out.npz"))
np.savez_compressed(filepath, **np_data)
print(total)
