using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class InfraredProcess : ScriptableRendererFeature
{

    [System.Serializable]
    public class InfraredProcessSettings
    {
        public Material material1;
        public Material material2;
        public bool IsInverse;
        public Texture OtherTex;
        public InfraredProcessSettings()
        {
            
        }
    }
    public InfraredProcessSettings settings = new InfraredProcessSettings();
    class InfraredRenderPass : ScriptableRenderPass
    {
        private Material material1;
        private Material material2;
        private bool IsInverse;
        private Texture OtherTex;

        public InfraredRenderPass(Material material1, Material material2, bool isInverse, Texture otherTex)
        {
            this.material1 = material1;
            this.material2 = material2;
            this.IsInverse = isInverse;
            this.OtherTex = otherTex;
        }
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
        }

        // Here you can implement the rendering logic.
        // Use <c>ScriptableRenderContext</c> to issue drawing commands or execute command buffers
        // https://docs.unity3d.com/ScriptReference/Rendering.ScriptableRenderContext.html
        // You don't have to call ScriptableRenderContext.submit, the render pipeline will call it at specific points in the pipeline.
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("InfraredProcess");
            RenderTargetIdentifier source = renderingData.cameraData.renderer.cameraColorTarget;
           
            if(IsInverse)
            {
                material1.SetTexture("_OtherTex", OtherTex);
                cmd.Blit(source, source, material1);
            }
            else
            {
                material2.SetTexture("_OtherTex", OtherTex);
                cmd.Blit(source, source, material2);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        // Cleanup any allocated resources that were created during the execution of this render pass.
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }
    }

    InfraredRenderPass m_ScriptablePass;

    /// <inheritdoc/>
    public override void Create()
    {
        m_ScriptablePass = new InfraredRenderPass(settings.material1,settings.material2,settings.IsInverse,settings.OtherTex);

        // Configures where the render pass should be injected.
        m_ScriptablePass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
    }

    // Here you can inject one or multiple render passes in the renderer.
    // This method is called when setting up the renderer once per-camera.
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_ScriptablePass);
    }
}


