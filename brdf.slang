// SPDX-License-Identifier: Apache-2.0
import "slangpy";
import sgl.device.print;

__exported import NeuralNetworks;

#define PI 3.14159265358979323846f
#define COMPONENT_WEIGHTS float4(1.f, 10.f, 1.f, 5.f)
#define NORMALIZED_COMPONENT_WEIGHTS (COMPONENT_WEIGHTS / dot(COMPONENT_WEIGHTS, 1.0f))

struct RNG
{
    uint state;

    __init(uint state)
    {
        this.state = state;
    }

    [mutating]
    uint next()
    {
        state = state * 2739110765U + 2739110765U;
        return state;
    }

    [mutating]
    float next1D()
    {
        // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
        // In floating-point precision this also ensures that 1.0-u != 0.0.
        uint bits = next();
        return (bits >> 8) * 0x1p-24;
    }

    [mutating]
    float2 next2D()
    {
        return { next1D(), next1D() };
    }
}

// Simple Camera struct to convert between screen space and world space
// coordinates, so that positions can be specified in world space.
struct Camera
{
    // World origin.
    float2 o;

    // World scale.
    float2 scale;

    // Frame dimensions (i.e. image resolution).
    float2 frameDim;

    // Convert from
    // screen coordinates [(0, 0), (W, H)] to
    // world coordinates [(o.x - scale.x, o.y - scale.y), (o.x + scale.x, o.y + scale.y)]

    [BackwardDifferentiable, NoDiffThis]
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }
};

// Simple struct to store the BRDF lighting properties.
struct Properties
{
    float3 baseColor;
    float roughness;
    float metallic;
    float specular;
};

float4 absDiff(float4 a, float4 b)
{
    return float4(abs(a.xyz - b.xyz), 1.0f);
}

float4 evalModel<Model : IModel<float[5], float4>>(Model model,
                                                   no_diff float NdotL, no_diff float NdotV, no_diff float NdotH, no_diff float LdotH, no_diff float roughness)
{
    return model.forward(float[5](NdotL, NdotV, NdotH, LdotH, roughness));
}

[BackwardDifferentiable]
float evalModelLoss<Model : IModel<float[5], float4>>(Model model,
                                                      no_diff float NdotL, no_diff float NdotV, no_diff float NdotH, no_diff float LdotH, no_diff float roughness,
                                                      no_diff float4 targetColor)

{
   
    let lossFunc = Losses::L2();

    float4 p = model.forward(float[5](NdotL, NdotV, NdotH, LdotH, roughness));
    float4 t = targetColor;

    return lossFunc.L2Relative(p, t);
}


void trainBRDF<Model : IModel<float[5], float4>>(Model model,
                                                 inout RNG rng, float lossScale, int2 sampleCoord, int2 batchDim, RWStructuredBuffer<float> lossOut)
{
    int idx = sampleCoord.y * batchDim.x + sampleCoord.x;

    // Using tangent coordinate system. N = (0,0,1)
    // L is arbitrary, but (N,L) >= 0 => L.z > 0, so generate random L in XZ plane's first quadrant
    float3 L;
    L.y = 0.f;
    sincos(rng.next1D() * PI / 2, L.z, L.x);

    // V is random direction, but (N,V) >= 0 => V.z > 0
    float sa, ca; // Azimuth [-PI/2, PI/2]
    sincos(-PI + 2 * PI * rng.next1D(), sa, ca);
    float se, ce;
    sincos(PI / 2 * rng.next1D(), se, ce);
    float3 V = float3(ce * ca, ce * sa, se);

    float NdotL = L.z;
    float NdotV = V.z;

    float3 H = normalize(L + V);
    float NdotH = H.z;
    float LdotH = dot(L, H);

    float roughness;
    if (rng.next1D() < 0.5f)
        roughness = rng.next1D();
    else
    {
        float u = rng.next1D();
        roughness = 1.0f - (u * u);
    }

    roughness = max(0.02f, roughness);

    float4 targetColor = DisneyBRDF(NdotL, NdotV, NdotH, LdotH, roughness);

    float loss = evalModelLoss(model,
                               NdotL, NdotV, NdotH, LdotH, roughness,
                               targetColor);
    lossOut[idx] = loss;

    bwd_diff(evalModelLoss)(model, NdotL, NdotV, NdotH, LdotH, roughness, targetColor, lossScale);
}

// Below are fairly standard BRDF lighting functions.
// Schlick Fresnel approximation for specular reflections.
[Differentiable]
float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

[Differentiable]
float Gtr1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / PI;
    }
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

[Differentiable]
float Gtr2(float NdotH, float ax)
{
    float a = ax * (1 / ax / ax * (1 - NdotH * NdotH) + NdotH * NdotH);
    return 1 / (PI * a * a);
}

[Differentiable]
float SmithGGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

[Differentiable]
float SmithGGXAnisotropy(float NdotV, float ax)
{
    return 1 / (NdotV + sqrt(ax * ax * (1 - NdotV * NdotV) + NdotV * NdotV));
}

[Differentiable]
float4 DisneyBRDF(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float ax = max(.001f, roughness * roughness);
    float Ds = Gtr2(NdotH, ax);
    float FH = SchlickFresnel(LdotH);
    float Gs = SmithGGXAnisotropy(NdotL, ax);
    Gs *= SmithGGXAnisotropy(NdotV, ax);

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, .25f) * SmithGGX(NdotV, .25f);

    return float4((1 / PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}

// BRDF function called from python, called for each pixel in the window. This
// tests if the pixelCoord is inside a sphere centered at world space (0,0) and
// then computes the lighting using BRDF for the given pixel. The z value is
// calculated using a ray-sphere intersection test to give a 3D sphere for
// better looking lighting and specular highlights.
float4 computeBRDF(Camera camera, Properties properties, int2 pixelCoord)
{
    // Default black background color.
    float4 outColor = float4(0.0, 0.0, 0.0, 0.0);

    float3 cameraPos = float3(0.0, 0.0, 2.0f);

    float3 lightDir = normalize(float3(-0.761f, -0.467f, -0.450f));
    float3 L = normalize(-lightDir);
    float3 lightIntensity = float3(1.0f, 1.0f, 1.0f);
    float3 baseColor = properties.baseColor;
    float roughness = properties.roughness;
    float metallic = properties.metallic;
    float specular = properties.specular;

    // Adjust pixel coordinate to have 0,0 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = camera.frameDim.y - fragCoord.y;

    // Convert the screen space fragCoord to world space.
    float2 screenSample = fragCoord.xy + 0.5;
    float2 worldSample = camera.screenToWorld(screenSample);

    // Create a sphere centered at world space origin (0,0). The x and y world
    // space coords have been calculated previously. The z coord is calculated
    // with a ray-sphere intersection test, treating the pixel itself as a ray
    // going through the sphere, giving a front and back z value.
    float front = 0, back = 0;
    bool hit = sphereIntersection(float3(0.0), 0.5, float3(worldSample.xy, 1.0), front, back);
    if (!hit)
        return outColor;

    float3 P = float3(worldSample, front);
    float3 V = normalize(cameraPos - P);
    float3 N = normalize(P);

    float3 H = normalize(L + V);
    float NdotL = max(0.f, dot(N, L));
    float NdotV = max(0.f, dot(N, V));
    float NdotH = max(0.f, dot(N, H));
    float LdotH = max(0.f, dot(L, H));


    float4 outParams = DisneyBRDF(NdotL, NdotV, NdotH, LdotH, roughness);

    // if (all(pixelCoord == DEBUG_PIX))
    // {

    //     print("NdotL={}, NdotV={}, NdotH={}, LdotH={}, roughness={}",
    //           NdotL, NdotV, NdotH, LdotH, roughness);
    //     print("ref params = {}", outParams.w);
    // }

    float3 Cdlin = pow(baseColor, 2.2f);
    float3 Cspec0 = lerp(specular * 0.08f.xxx, Cdlin, metallic);

    float3 brdfn =
        outParams.x * Cdlin * (1.0f - metallic) +
        outParams.y * lerp(Cspec0, 1.0f.xxx, outParams.z) +
        outParams.w;

    float3 color = brdfn * NdotL * lightIntensity;

    return float4(color, 1.0f);
}

static const int2 DEBUG_PIX = int2(444, 308);

// BRDF function called from python, called for each pixel in the window. This
// tests if the pixelCoord is inside a sphere centered at world space (0,0) and
// then computes the lighting using BRDF for the given pixel. The z value is
// calculated using a ray-sphere intersection test to give a 3D sphere for
// better looking lighting and specular highlights.
float4 computeBRDFNeural<Model : IModel<float[5], float4>>(Model model, Camera camera, Properties properties, int2 pixelCoord)
{
    // Default black background color.
    float4 outColor = float4(0.0, 0.0, 0.0, 0.0);

    float3 cameraPos = float3(0.0, 0.0, 2.0f);

    float3 lightDir = normalize(float3(-0.761f, -0.467f, -0.450f));
    float3 L = normalize(-lightDir);
    float3 lightIntensity = float3(1.0f, 1.0f, 1.0f);
    float3 baseColor = properties.baseColor;
    float roughness = properties.roughness;
    float metallic = properties.metallic;
    float specular = properties.specular;

    // Adjust pixel coordinate to have 0,0 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = camera.frameDim.y - fragCoord.y;

    // Convert the screen space fragCoord to world space.
    float2 screenSample = fragCoord.xy + 0.5;
    float2 worldSample = camera.screenToWorld(screenSample);

    // Create a sphere centered at world space origin (0,0). The x and y world
    // space coords have been calculated previously. The z coord is calculated
    // with a ray-sphere intersection test, treating the pixel itself as a ray
    // going through the sphere, giving a front and back z value.
    float front = 0, back = 0;
    bool hit = sphereIntersection(float3(0.0), 0.5, float3(worldSample.xy, 1.0), front, back);

    if (!hit)
        return outColor;

    float3 P = float3(worldSample, front);
    float3 V = normalize(cameraPos - P);
    float3 N = normalize(P);

    float3 H = normalize(L + V);
    float NdotL = max(0.f, dot(N, L));
    float NdotV = max(0.f, dot(N, V));
    float NdotH = max(0.f, dot(N, H));
    float LdotH = max(0.f, dot(L, H));

    float4 outParams = evalModel(model, NdotL, NdotV, NdotH, LdotH, roughness);
    float4 outParamsRef = DisneyBRDF(NdotL, NdotV, NdotH, LdotH, roughness);

    // if (all(pixelCoord == DEBUG_PIX))
    // {
    //     float4 diff = outParams - outParamsRef;
    //     float4 absd = abs(diff);

    //     print("DEBUG pix=({}, {})", pixelCoord.x, pixelCoord.y);
    //     print("inputs: NdotL={} NdotV={} NdotH={} LdotH={} rough={}",
    //           NdotL, NdotV, NdotH, LdotH, roughness);

    //     print("params nn  = [{}, {}, {}, {}]", outParams.x, outParams.y, outParams.z, outParams.w);
    //     print("params ref = [{}, {}, {}, {}]", outParamsRef.x, outParamsRef.y, outParamsRef.z, outParamsRef.w);
    //     print("abs diff   = [{}, {}, {}, {}]", absd.x, absd.y, absd.z, absd.w);
    // }

    float3 Cdlin = pow(baseColor, 2.2f);
    float3 Cspec0 = lerp(specular * 0.08f.xxx, Cdlin, metallic);

    float3 brdfn =
        outParams.x * Cdlin * (1.0f - metallic) +
        outParams.y * lerp(Cspec0, 1.0f.xxx, outParams.z) +
        outParams.w;

    float3 color = brdfn * NdotL * lightIntensity;

    return float4(color, 1.0f);
}

// Below are ray-sphere intersection test equations.

// The quadratic equation is not numerically stable with limited precision
// floating point values, so we use a more numerically stable version to
// get the results from the quadratic equation.
[Differentiable]
float2 getDepthDistances(float4 eq)
{
    float discr = sqrt(eq.w);
    float d1, d2, z1, z2;

    if (eq.y >= 0)
    {
        d1 = 2.0 * eq.x;
        d2 = -eq.y - discr;
        if (d1 == 0.0)
            z1 = 0.0;
        else
            z1 = (-eq.y - discr) / d1;
        if (d2 == 0.0)
            z2 = 0.0;
        else
            z2 = (2.0 * eq.z) / d2;
    }
    else
    {
        d1 = -eq.y + discr;
        d2 = 2.0 * eq.x;
        if (d1 == 0.0)
            z1 = 0.0;
        else
            z1 = (2.0 * eq.z) / d1;
        if (d2 == 0.0)
            z2 = 0.0;
        else
            z2 = (-eq.y + discr) / d2;
    }

    return float2(z1, z2);
}

// Get the front and back z values using the sphere equation results and
// ray direction.
[Differentiable]
float2 getDepths(float4 eq, float3 dir)
{
    float2 z = getDepthDistances(eq);
    return float2((dir * z.x).z, (dir * z.y).z);
}

// Simple quadratic equation for the sphere.
[Differentiable]
float4 sphereEquation(float3 spherePos, float radius, float3 ray, float3 origin)
{
    float3 l = normalize(ray);
    float3 p = spherePos;
    float r2 = radius * radius;

    float a = dot(l, l);
    float b = 2.0 * dot(l, (origin - p));
    float c = dot((origin - p), (origin - p)) - r2;
    float discr = (b * b) - (4.0 * a * c);

    return float4(a, b, c, discr);
}

// Perform a ray-sphere intersection test for a given sphere and radius. This
// returns true if the result is inside the sphere, and computes front and
// back depth values for the intersection. Note that this test is in world
// space due to the world space camera position (0,0,1). This is somewhat
// hacky and would be better in eye space (0,0,0) but currently left in world
// space for somplicity.
[Differentiable]
bool sphereIntersection(float3 spherePos, float radius, float3 ray,
                        out float frontDepth, out float backDepth)
{
    // World space origin of the ray is at (0,0,1).
    float4 eq = sphereEquation(spherePos, radius, ray, float3(0.0, 0.0, 1.0));

    frontDepth = 0;
    backDepth = 0;

    // No intersections.
    if (eq.w < 0)
    {
        return false;
    }

    float2 depths = getDepths(eq, ray);

    frontDepth = min(-depths.x, -depths.y);
    backDepth = max(-depths.x, -depths.y);

    return true;
}
