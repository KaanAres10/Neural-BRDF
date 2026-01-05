// SPDX-License-Identifier: Apache-2.0

implementing NeuralNetworks;

namespace Losses
{

public struct L2
{
    public __init()
    {
    }

    [BackwardDifferentiable]
    public float eval(float3 pred, float3 ref)
    {
        float3 diff = (pred - ref);
        diff *= diff;
        return (diff[0] + diff[1] + diff[2]) / 3.f;
    }

    [BackwardDifferentiable]
    public float eval(float4 pred, float4 ref)
    {
        float4 diff = (pred - ref);
        diff *= diff;
        return (diff[0] + diff[1] + diff[2] + diff[3]) / 4.f;
    }

    [BackwardDifferentiable]
    public float L2Relative(float4 pred, float4 ref)
    {
        float4 diff = pred - ref;
        float l2err = sqrt(dot(diff, diff));
        float l2ref = sqrt(dot(ref, ref));
        return l2err / (l2ref + 1e-6f);
    }

    [BackwardDifferentiable]
    public float L2RelativeWeighted(float4 pred, float4 ref, float4 w)
    {
        float4 d = pred - ref;

        // weighted squared norms
        float err2 = dot(w, d * d);
        float ref2 = dot(w, ref * ref);

        float l2err = sqrt(err2);
        float l2ref = sqrt(ref2);

        return l2err / (l2ref + 1e-6f);
    }
}

}
