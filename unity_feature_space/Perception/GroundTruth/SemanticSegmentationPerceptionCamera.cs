using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.Universal;
using System;

namespace UnityEngine.Perception.GroundTruth
{
    public class SemanticSegmentationPerceptionCamera : PerceptionCamera
    {
        public SemanticSegmentationLabelConfig SemanticSegmentationLabelConfig;

        RenderTexture m_SemanticSegmentationTexture;
        Texture2D m_CpuTexture;

        Camera m_Camera;
        int m_LastFrameEndRendering = -1;
        public event Action RenderedObjectInfosCalculated;
        public RenderTexture SemanticSegmentationTexture => m_SemanticSegmentationTexture;

        public void Start()
        {
            Camera camera = GetComponent<Camera>();
            m_Camera = camera;

            var width = m_Camera.pixelWidth;
            var height = m_Camera.pixelHeight;

            m_SemanticSegmentationTexture = new RenderTexture(width, height, 8, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            m_SemanticSegmentationTexture.Create();
            m_SemanticSegmentationTexture.name = "SemanticSegmentation";

            if (TestRawImage != null)
                TestRawImage.texture = m_SemanticSegmentationTexture;

            SemanticSegmentationUrpPass semanticSegmentationPass = new SemanticSegmentationUrpPass(m_Camera, m_SemanticSegmentationTexture, SemanticSegmentationLabelConfig);
            passes.Add(semanticSegmentationPass);

            LensDistortionUrpPass lensDistortionPass = new LensDistortionUrpPass(m_Camera, m_SemanticSegmentationTexture);
            passes.Add(lensDistortionPass);

            RenderPipelineManager.endFrameRendering += OnEndFrameRendering;
        }

        void OnEndFrameRendering(ScriptableRenderContext scriptableRenderContext, Camera[] cameras)
        {
            if (Application.isPlaying)
            {
                if (ManuallyCapture && !ShouldCapture || !ManuallyCapture && m_LastFrameEndRendering == Time.frameCount)
                    return;

                ShouldCapture = false;
                m_LastFrameEndRendering = Time.frameCount;

                if (!cameras.Any(c => c == m_Camera))
                    return;

                RenderedObjectInfosCalculated?.Invoke();
            }
        }
    }
}