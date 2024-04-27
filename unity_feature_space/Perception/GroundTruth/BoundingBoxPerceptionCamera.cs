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
    public struct BoundingBoxInfo : IEquatable<BoundingBoxInfo>
    {
        public int labelId;
        public string label;
        public Rect boundingBox;
        public int pixelCount;
        public Color32 labelIdColor;

        /// <inheritdoc />
        public override string ToString()
        {
            return $"{nameof(labelId)}: {labelId}, {nameof(boundingBox)}: {boundingBox}, " +
                $"{nameof(pixelCount)}: {pixelCount}, {nameof(labelIdColor)}: {labelIdColor}";
        }

        /// <inheritdoc />
        public bool Equals(BoundingBoxInfo other)
        {
            return labelId == other.labelId &&
                boundingBox.Equals(other.boundingBox) &&
                pixelCount == other.pixelCount;
        }

        /// <inheritdoc />
        public override bool Equals(object obj)
        {
            return obj is BoundingBoxInfo other && Equals(other);
        }

        /// <inheritdoc />
        public override int GetHashCode()
        {
            unchecked
            {
                // ReSharper disable NonReadonlyMemberInGetHashCode
                var hashCode = (int)labelId;
                hashCode = (hashCode * 397) ^ boundingBox.GetHashCode();
                hashCode = (hashCode * 397) ^ pixelCount;
                return hashCode;
            }
        }
    }

    public class BoundingBoxPerceptionCamera : PerceptionCamera
    {
        public IdLabelConfig IdLabelConfig;
        public SemanticSegmentationLabelConfig SemanticSegmentationLabelConfig;


        RenderedObjectInfoGenerator m_RenderedObjectInfoGenerator;
        RenderTexture m_InstanceSegmentationTexture;
        Texture2D m_CpuTexture;

        Camera m_Camera;
        int m_LastFrameEndRendering = -1;
        public event Action<List<BoundingBoxInfo>> RenderedObjectInfosCalculated;

        public void Start()
        {
            Camera camera = GetComponent<Camera>();
            m_Camera = camera;
            var width = camera.pixelWidth;
            var height = camera.pixelHeight;

            m_RenderedObjectInfoGenerator = new RenderedObjectInfoGenerator();

            m_InstanceSegmentationTexture = new RenderTexture(new RenderTextureDescriptor(width, height, GraphicsFormat.R8G8B8A8_UNorm, 8));
            m_InstanceSegmentationTexture.filterMode = FilterMode.Point;
            m_InstanceSegmentationTexture.name = "InstanceSegmentation";
            m_InstanceSegmentationTexture.Create();

            if (TestRawImage != null)
                TestRawImage.texture = m_InstanceSegmentationTexture;

            InstanceSegmentationUrpPass instanceSegmentationPass = new InstanceSegmentationUrpPass(camera, m_InstanceSegmentationTexture);
            passes.Add(instanceSegmentationPass);

            LensDistortionUrpPass lensDistortionPass = new LensDistortionUrpPass(camera, m_InstanceSegmentationTexture);
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

                var width = m_InstanceSegmentationTexture.width;
                var height = m_InstanceSegmentationTexture.height;

                var oriRenderTexture = RenderTexture.active;
                RenderTexture.active = m_InstanceSegmentationTexture;

                if (m_CpuTexture == null)
                    m_CpuTexture = new Texture2D(width, height, m_InstanceSegmentationTexture.graphicsFormat, TextureCreationFlags.None);

                m_CpuTexture.ReadPixels(new Rect(Vector2.zero, new Vector2(width, height)), 0, 0);
                RenderTexture.active = oriRenderTexture;

                var data = m_CpuTexture.GetRawTextureData<Color32>();

                m_RenderedObjectInfoGenerator.Compute(data, width,
                        BoundingBoxOrigin.TopLeft, out var renderedObjectInfos, Allocator.Temp);

                var boundingBoxInfos = new List<BoundingBoxInfo>();

                for (var i = 0; i < renderedObjectInfos.Length; i++)
                {
                    var info = renderedObjectInfos[i];
                    IdLabelEntry idLabelEntry;
                    IdLabelConfig.TryGetLabelEntryFromInstanceId(info.instanceId, out idLabelEntry);
                    var semanticSegamentationLabelEntries = SemanticSegmentationLabelConfig.labelEntries.Where(p => p.label == idLabelEntry.label).ToList();
                    if (semanticSegamentationLabelEntries.Count > 0)
                    {
                        var semanticSegmentationLabelEntry = semanticSegamentationLabelEntries[0];

                        boundingBoxInfos.Add(new BoundingBoxInfo
                        {
                            labelId = idLabelEntry.id,
                            label = idLabelEntry.label,
                            boundingBox = info.boundingBox,
                            pixelCount = info.pixelCount,
                            labelIdColor = semanticSegmentationLabelEntry.color
                        });
                    }
                }

                RenderedObjectInfosCalculated?.Invoke(boundingBoxInfos);

                renderedObjectInfos.Dispose();
            }
        }
    }
}