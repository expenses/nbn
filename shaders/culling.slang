
struct CullingInfo {
    float4x4 view;
    float4x4 perspective;
    float2 near_plane;
};

bool cull_bounding_sphere(CullingInfo info, float4 bounding_sphere) {
    float3 world_space_pos = bounding_sphere.xyz;
    float radius = bounding_sphere.w;

    float3 view_space_pos =
        mul(info.view, float4(world_space_pos, 1.0)).xyz;
    // why?
    view_space_pos.z = -view_space_pos.z;

    // Is the most positive/forwards point of the object in front of the near plane?
    bool visible = view_space_pos.z + radius > info.near_plane.x;

    // Do some fancy stuff by getting the frustum planes and comparing the position against them.
    float3 frustum_x = normalize(
        info.perspective[3].xyz
        + transpose(info.perspective)[0].xyz
    );
    float3 frustum_y = normalize(
        transpose(info.perspective)[3].xyz
        + transpose(info.perspective)[1].xyz
    );

    //visible = visible
    //    && view_space_pos.z * frustum_x.z + abs(view_space_pos.x) * frustum_x.x
    //        < radius;
    //visible = visible
    //    && view_space_pos.z * frustum_y.z - abs(view_space_pos.y) * frustum_y.y
    //        < radius;

    return !visible;
}
