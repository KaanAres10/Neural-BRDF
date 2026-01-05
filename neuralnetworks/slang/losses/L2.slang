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
    public float L2Relative(float4 pred, float4 target, float4 scale)
    {
        float4 diff = pred - target;
        float4 denom = pred * pred + 0.01f.xxxx;
        float4 v = scale * (diff * diff) / denom;
        return v.x + v.y + v.z + v.w;
    }

    

}

}
