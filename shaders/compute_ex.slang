import compute_ex_structs;

[shader("compute")]
[NumThreads(64, 1, 1)]
void add(uint thread: SV_DispatchThreadID,
    uniform PushConstants pc) {

if (thread >= pc.num_items) {
return;
}

pc.buffer[thread]+=1.7;
}
