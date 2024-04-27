Shader "Hidden/Thermal1"
{
    Properties
    {
        _MainTex ("Base (RGB)", 2D) = "white" {}
        _OtherTex("Other (RGB)", 2D) = "white" {}
    }
        SubShader
    {
        Pass {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"
            uniform sampler2D _MainTex;
            uniform sampler2D _OtherTex;
            float4 frag(v2f_img i) : COLOR{
                fixed4 renderTex = tex2D(_MainTex, i.uv);
                fixed4 OtherTex = tex2D(_OtherTex, i.uv);

                fixed gray1 = 0.2125 * renderTex.r + 0.7154 * renderTex.g + renderTex.b;
                fixed gray2 = 0.2125 * OtherTex.r + 0.7154 * OtherTex.g + OtherTex.b;

                fixed maxgr = gray1 * 0.1 * step(gray2, gray1) + gray2 * step(gray1, gray2);
                //fixed maxgr = max(gray1, gray2);
                fixed3 grayColor = float3(maxgr, maxgr, maxgr);

                float4 result = renderTex;
                result.rgb = grayColor;
                return result;
            }
            ENDCG
    
        }
        
    }
}
