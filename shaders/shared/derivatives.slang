
public struct WithDerivatives<T> {
    public T val;
    public T dx;
    public T dy;
};

func gen_derivatives<T, let N : int>(vector<T, N> val)
    -> WithDerivatives<vector<T, N>> where T: __BuiltinFloatingPointType {
    WithDerivatives<vector<T, N>> out;
    out.val = val;
    out.dx = ddx(val);
    out.dy = ddy(val);
    return out;
}
