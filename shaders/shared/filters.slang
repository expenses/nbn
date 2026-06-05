// Adapted from https://github.com/TheRealMJP/MSAAFilter/blob/cfc1c19411e24dc07f74272c17f423330ac96941/SampleFramework11/v1.01/Graphics/Filtering.h#L101
// MIT

inline float FilterCubic1D(float x, float B, float C)
{
    // Rescale from [-1, 1] range to [-2, 2]
    x = abs(x) * 2.0f;

    float y = 0.0f;
    float x2 = x * x;
    float x3 = x * x * x;
    if(x < 1)
        y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
    else if(x <= 2)
        y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

    return y / 6.0f;
}

inline float FilterCubic2D(float2 xy, float B, float C)
{
    return FilterCubic1D(xy.x, B, C) * FilterCubic1D(xy.y, B, C);
}

inline float FilterBSpline1D(float x)
{
    return FilterCubic1D(x, 1.0f, 0.0f);
}

inline float FilterBSpline2D(float2 xy)
{
    return FilterBSpline1D(xy.x) * FilterBSpline1D(xy.y);
}

inline float FilterCatmullRom1D(float x)
{
    return FilterCubic1D(x, 0.0f, 0.5f);
}

inline float FilterCatmullRom2D(float2 xy)
{
    return FilterCatmullRom1D(xy.x) * FilterCatmullRom1D(xy.y);
}

inline float FilterMitchell1D(float x)
{
    return FilterCubic1D(x, 1 / 3.0f, 1 / 3.0f);
}

inline float FilterMitchell2D(float2 xy)
{
    return FilterMitchell1D(xy.x) * FilterMitchell1D(xy.y);
}
