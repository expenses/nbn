import derivatives;
import util;

// Copied from
// http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/.
//
// by way of
// https://github.com/expenses/lighthugger/blob/main/src/shaders/common/vbuffer.glsl

struct BarycentricDeriv {
    float3 m_lambda;
    float3 m_ddx;
    float3 m_ddy;
};

struct BarycentricsAndInterpolatedPosition {
    BarycentricDeriv barycentrics;
    WithDerivatives<float3> position;

}

BarycentricsAndInterpolatedPosition
calculate_barycentrics_from_world_positions(
    float4x4 mat,
    float3 pos_a,
    float3 pos_b,
    float3 pos_c,
    uint2 thread,
    uint2 extent
) {
    BarycentricsAndInterpolatedPosition output;
    output.barycentrics = CalcFullBary(
        mul(mat, float4(pos_a, 1.0)),
        mul(mat, float4(pos_b, 1.0)),
        mul(mat, float4(pos_c, 1.0)),
        thread_to_clip(thread, extent),
        extent
    );
    output.position = interpolate(output.barycentrics, pos_a, pos_b, pos_c);
    return output;
}

BarycentricDeriv CalcFullBary(
    float4 pt0,
    float4 pt1,
    float4 pt2,
    float2 pixelNdc,
    float2 winSize
) {
    // EDIT: got rid of the cast.
    BarycentricDeriv ret;

    float3 invW = rcp(float3(pt0.w, pt1.w, pt2.w));

    float2 ndc0 = pt0.xy * invW.x;
    float2 ndc1 = pt1.xy * invW.y;
    float2 ndc2 = pt2.xy * invW.z;

    float invDet = rcp(determinant(float2x2(ndc2 - ndc1, ndc0 - ndc1)));
    ret.m_ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y)
        * invDet * invW;
    ret.m_ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x)
        * invDet * invW;
    float ddxSum = dot(ret.m_ddx, float3(1, 1, 1));
    float ddySum = dot(ret.m_ddy, float3(1, 1, 1));

    float2 deltaVec = pixelNdc - ndc0;
    float interpInvW = invW.x + deltaVec.x * ddxSum + deltaVec.y * ddySum;
    float interpW = rcp(interpInvW);

    ret.m_lambda.x = interpW
        * (invW[0] + deltaVec.x * ret.m_ddx.x + deltaVec.y * ret.m_ddy.x);
    ret.m_lambda.y =
        interpW * (0.0f + deltaVec.x * ret.m_ddx.y + deltaVec.y * ret.m_ddy.y);
    ret.m_lambda.z =
        interpW * (0.0f + deltaVec.x * ret.m_ddx.z + deltaVec.y * ret.m_ddy.z);

    ret.m_ddx *= (2.0f / winSize.x);
    ret.m_ddy *= (2.0f / winSize.y);
    ddxSum *= (2.0f / winSize.x);
    ddySum *= (2.0f / winSize.y);

    ret.m_ddy *= -1.0f;
    ddySum *= -1.0f;

    float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    float interpW_ddy = 1.0f / (interpInvW + ddySum);

    ret.m_ddx =
        interpW_ddx * (ret.m_lambda * interpInvW + ret.m_ddx) - ret.m_lambda;
    ret.m_ddy =
        interpW_ddy * (ret.m_lambda * interpInvW + ret.m_ddy) - ret.m_lambda;

    return ret;
}

float3
InterpolateWithDeriv(BarycentricDeriv deriv, float v0, float v1, float v2) {
    float3 mergedV = float3(v0, v1, v2);
    float3 ret;
    ret.x = dot(mergedV, deriv.m_lambda);
    ret.y = dot(mergedV, deriv.m_ddx);
    ret.z = dot(mergedV, deriv.m_ddy);
    return ret;
}

// Helper structs.

WithDerivatives<float2>
interpolate(BarycentricDeriv deriv, float2 v0, float2 v1, float2 v2) {
    WithDerivatives<float2> interp;
    float3 x = InterpolateWithDeriv(deriv, v0.x, v1.x, v2.x);
    interp.val.x = x.x;
    interp.dx.x = x.y;
    interp.dy.x = x.z;
    float3 y = InterpolateWithDeriv(deriv, v0.y, v1.y, v2.y);
    interp.val.y = y.x;
    interp.dx.y = y.y;
    interp.dy.y = y.z;
    return interp;
}

WithDerivatives<float3>
interpolate(BarycentricDeriv deriv, float3 v0, float3 v1, float3 v2) {
    WithDerivatives<float3> interp;
    float3 x = InterpolateWithDeriv(deriv, v0.x, v1.x, v2.x);
    interp.val.x = x.x;
    interp.dx.x = x.y;
    interp.dy.x = x.z;
    float3 y = InterpolateWithDeriv(deriv, v0.y, v1.y, v2.y);
    interp.val.y = y.x;
    interp.dx.y = y.y;
    interp.dy.y = y.z;
    float3 z = InterpolateWithDeriv(deriv, v0.z, v1.z, v2.z);
    interp.val.z = z.x;
    interp.dx.z = z.y;
    interp.dy.z = z.z;
    return interp;
}

func interpolate_ray_barycentrics<int N>(
    vector<float, N> a,
    vector<float, N> b,
    vector<float, N> c,
    float2 barycentrics
) -> vector<float, N> {
    return a * (1.0 - barycentrics.x - barycentrics.y) + b * barycentrics.x
        + c * barycentrics.y;
}
