using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.UI;

namespace Unity.MLAgents.Sensors
{
    public class SemanticSegmentationPerceptionRenderTextureSensor : ISensor
    {
        RenderTexture m_RenderTexture;
        SemanticSegmentationPerceptionCamera m_PerceptionCamera;
        float m_Random;

        // 使用RenderTextureSensor作为基础sensor，外层套PerceptionCamera的壳
        RenderTextureSensor m_RenderTextureSensor;

        public SemanticSegmentationPerceptionRenderTextureSensor(
            int width, int height,
            SemanticSegmentationPerceptionCamera perceptionCamera,
            float random,
            bool grayscale, string name, SensorCompressionType compressionType)
        {
            m_RenderTexture = new RenderTexture(width, height, 8, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            m_PerceptionCamera = perceptionCamera;
            m_Random = random;

            m_PerceptionCamera.GetComponent<Camera>().targetTexture = RenderTexture.GetTemporary(width, height, 8, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            m_PerceptionCamera.ManuallyCapture = true;
            m_PerceptionCamera.RenderedObjectInfosCalculated += RenderedObjectInfosCalculated;
            LabelManager.singleton.RegisterPendingLabels();

            m_RenderTextureSensor = new RenderTextureSensor(m_RenderTexture, grayscale, name, compressionType);
        }

        private void RenderedObjectInfosCalculated()
        {
            var oriRenderTexture = RenderTexture.active;
            RenderTexture.active = m_RenderTexture;
            Graphics.Blit(m_PerceptionCamera.SemanticSegmentationTexture, m_RenderTexture);
            RenderTexture.active = oriRenderTexture;
        }

        public RenderTexture RenderTexture
        {
            get { return m_RenderTexture; }
        }

        public void SetRandom(float random)
        {
            m_Random = random;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_RenderTextureSensor.GetName();
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_RenderTextureSensor.GetObservationSpec();
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            return m_RenderTextureSensor.GetCompressedObservation();
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            return m_RenderTextureSensor.Write(writer);
        }

        /// <inheritdoc/>
        public void Update()
        {
            LabelManager.singleton.RegisterPendingLabels();

            var registeredLabels = LabelManager.singleton.registeredLabels.ToList();
            List<Vector3> tmpPosition = new List<Vector3>(registeredLabels.Count);

            if (m_Random != 0)
            {
                registeredLabels.ForEach(label =>
                            {
                                tmpPosition.Add(label.transform.position);
                                Vector3 randomDelta = new Vector3(Random.Range(0, m_Random ),
                                                                    Random.Range(0, m_Random),
                                                                    Random.Range(0, m_Random ));
                                label.transform.position += randomDelta;
                            });
            }

            m_PerceptionCamera.ShouldCapture = true;
            m_PerceptionCamera.GetComponent<Camera>().Render();

            if (m_Random != 0)
            {
                for (int i = 0; i < registeredLabels.Count; i++)
                {
                    var label = registeredLabels[i];
                    label.transform.position = tmpPosition[i];
                }
            }
        }

        /// <inheritdoc/>
        public void Reset()
        {
        }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return m_RenderTextureSensor.GetCompressionSpec();
        }

        public void Dispose()
        {
            m_PerceptionCamera.RenderedObjectInfosCalculated -= RenderedObjectInfosCalculated;
        }
    }

    public class SemanticSegmentationComponent : SensorComponent
    {
        [Tooltip("挂载了`SemanticSegmentationPerceptionCamera`的摄像机，置空则为当前物体")]
        public SemanticSegmentationPerceptionCamera SemanticSegmentationPerceptionCamera;

        [SerializeField]
        private float m_Random = 0;
        public float Random
        {
            get => m_Random;
            set { m_Random = value; UpdateSensor(); }
        }

        public string SensorName = "SegmentationSensor";
        public bool Grayscale = false;
        public SensorCompressionType Compression = SensorCompressionType.PNG;

        public Vector2Int ImageSize = new Vector2Int(200, 200);
        public RawImage SegmentationRawImage;

        SemanticSegmentationPerceptionRenderTextureSensor m_Sensor;

        public override ISensor[] CreateSensors()
        {
            if (m_Sensor != null)
                return new ISensor[] { m_Sensor };

            var camera = SemanticSegmentationPerceptionCamera;
            if (camera == null)
            {
                camera = GetComponent<SemanticSegmentationPerceptionCamera>();
            }

            m_Sensor = new SemanticSegmentationPerceptionRenderTextureSensor(
                ImageSize.x, ImageSize.y,
                camera,
                Random,
                Grayscale, SensorName, Compression);

            if (SegmentationRawImage && Application.isPlaying)
            {
                SegmentationRawImage.texture = m_Sensor.RenderTexture;
            }

            return new ISensor[] { m_Sensor };
        }

        private void UpdateSensor()
        {
            if (m_Sensor != null)
            {
                m_Sensor.SetRandom(m_Random);
            }
        }
    }
}