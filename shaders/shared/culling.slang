bool coneCull(
    float3 center,
    float radius,
    float3 cone_axis,
    float cone_cutoff,
    float3 camera_position
) {
    return dot(normalize(center - camera_position), cone_axis)
        <= cone_cutoff + radius / length(center - camera_position);
}

bool frustumCull(float3 view_center, float radius, float near_plane, float4 frustum) {
    // near plane
    var visible = view_center.z - radius < near_plane;
    // left and right planes
    visible &= view_center.z * frustum[1]
            - abs(view_center.x) * frustum[0]
        > -radius;
    // top and bottom planes
    visible &= view_center.z * frustum[3]
            + abs(view_center.y) * frustum[2]
        > -radius;
    return visible;
}
