
[numthreads(64,1,1)]
void main(uint8_t* output, uint3 dispatch_thread_id : SV_DispatchThreadID,
    )
{
    output[dispatch_thread_id.x] = 11;
}
