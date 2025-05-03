import ffx_shadows;

[numthreads(8, 8, 1)]
void filter_pass_0(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID)
{
    DenoiserShadowsFilterPass0(gid, gtid, did);
}

[numthreads(8, 8, 1)]
void filter_pass_1(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID)
{
    DenoiserShadowsFilterPass1(gid, gtid, did);
}

[numthreads(8, 8, 1)]
void filter_pass_2(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID)
{
    DenoiserShadowsFilterPass2(gid, gtid, did);
}
