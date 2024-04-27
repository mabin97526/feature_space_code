using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Linq;

namespace Unity.MLAgents.Sensors
{
    public class BoundingBoxPerceptionSensor : ISensor
    {
        int m_maxNumberObject;
        int m_numberObjectType;

        int m_Width;
        int m_Height;
        BoundingBoxPerceptionCamera m_PerceptionCamera;
        float m_Random;

        BufferSensor m_BufferSensor;

        /// <summary>
        /// BoundingBox 传感器，基于 BufferSensor
        /// </summary>
        /// <param name="maxNumberObject">可识别的最多的物体数</param>
        /// <param name="numberObjectType">最大的物体类别数</param>
        /// <param name="width">摄像机的宽度</param>
        /// <param name="height">摄像机的高度</param>
        /// <param name="perceptionCamera">感知摄像机</param>
        /// <param name="random">对识别物体bbox的扰动</param>
        /// <param name="name">传感器名字</param>
        public BoundingBoxPerceptionSensor(
            int maxNumberObject,
            int numberObjectType,

            int width, int height,
            BoundingBoxPerceptionCamera perceptionCamera,
            float random,
            string name)
        {
            m_maxNumberObject = maxNumberObject;
            m_numberObjectType = numberObjectType;

            m_Width = width;
            m_Height = height;
            m_PerceptionCamera = perceptionCamera;
            m_Random = random;

            m_PerceptionCamera.GetComponent<Camera>().targetTexture = RenderTexture.GetTemporary(width, height, 24); // 强制按照[width height]进行渲染
            m_PerceptionCamera.ManuallyCapture = true;
            m_PerceptionCamera.RenderedObjectInfosCalculated += RenderedObjectInfosCalculated;
            LabelManager.singleton.RegisterPendingLabels();

            m_BufferSensor = new BufferSensor(maxNumberObject, numberObjectType + 4, name);
        }

        private void RenderedObjectInfosCalculated(List<BoundingBoxInfo> boundingBoxInfos)
        {
            foreach (var boundingBoxInfo in boundingBoxInfos)
            {
                if (Random.value < m_Random) // 随机去除某些bbox
                    continue;

                var boundingBox = boundingBoxInfo.boundingBox;
                var x = boundingBox.x / m_Width + Random.Range(0, m_Random);
                var y = boundingBox.y / m_Height + Random.Range(0, m_Random);
                var xMax = boundingBox.xMax / m_Width + Random.Range(0, m_Random);
                var yMax = boundingBox.yMax / m_Height + Random.Range(0, m_Random);

                x = Mathf.Clamp(x, 0, 1);
                y = Mathf.Clamp(y, 0, 1);
                xMax = Mathf.Clamp(xMax, 0, 1);
                yMax = Mathf.Clamp(yMax, 0, 1);

                var obs = new float[m_numberObjectType + 4];
                obs[boundingBoxInfo.labelId] = 1;

                // 以图像左上角为原点(0,0)，横坐标x，纵坐标y，记录bbox的中心点与长宽
                // 中心点范围 [0, 1], 长宽范围 [0, 1]
                obs[m_numberObjectType + 0] = (xMax - x) / 2 + x;
                obs[m_numberObjectType + 1] = (yMax - y) / 2 + y;
                obs[m_numberObjectType + 2] = xMax - x;
                obs[m_numberObjectType + 3] = yMax - y;
                //UnityEngine.Debug.Log(obs[m_numberObjectType + 3]);
                m_BufferSensor.AppendObservation(obs);
            }
        }

        public void SetRandom(float random)
        {
            m_Random = random;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_BufferSensor.GetName();
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_BufferSensor.GetObservationSpec();
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            return m_BufferSensor.GetCompressedObservation();
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            return m_BufferSensor.Write(writer);
        }

        /// <inheritdoc/>
        public void Update()
        {
            m_BufferSensor.Update();

            LabelManager.singleton.RegisterPendingLabels();

            m_PerceptionCamera.ShouldCapture = true;
            m_PerceptionCamera.GetComponent<Camera>().Render();
        }

        /// <inheritdoc/>
        public void Reset()
        {
            m_BufferSensor.Reset();
        }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return m_BufferSensor.GetCompressionSpec();
        }

        public void Dispose()
        {
            m_PerceptionCamera.RenderedObjectInfosCalculated -= RenderedObjectInfosCalculated;
        }
    }

    [System.Serializable]
    public struct LabelColor
    {
        public string Label;
        public Color Color;
    }

    public class BoundingBoxComponent : SensorComponent
    {
        [Tooltip("挂载了`BoundingBoxPerceptionCamera`的摄像机，置空则为当前物体")]
        public BoundingBoxPerceptionCamera BoundingBoxPerceptionCamera;

        [Tooltip("可识别的最多的物体数")]
        public int MaxNumberObject;

        [Tooltip("最大的物体类别数")]
        public int NumberObjectType;

        [SerializeField, Range(0f, 1f)]
        private float m_Random = 0;
        public float Random
        {
            get => m_Random;
            set { m_Random = value; UpdateSensor(); }
        }

        public string SensorName = "BoundingBoxSensor";

        public Vector2Int ImageSize = new Vector2Int(200, 200);

        BoundingBoxPerceptionSensor m_Sensor;

        public override ISensor[] CreateSensors()
        {
            if (m_Sensor != null)
                return new ISensor[] { m_Sensor };

            Dispose();

            var camera = BoundingBoxPerceptionCamera;
            if (camera == null)
            {
                camera = GetComponent<BoundingBoxPerceptionCamera>();
            }

            m_Sensor = new BoundingBoxPerceptionSensor(
                MaxNumberObject,
                NumberObjectType,

                ImageSize.x, ImageSize.y,
                camera,
                Random,
                SensorName);

            return new ISensor[] { m_Sensor };
        }

        private void UpdateSensor()
        {
            if (m_Sensor != null)
            {
                m_Sensor.SetRandom(m_Random);
            }
        }

        public void Dispose()
        {
            if (!ReferenceEquals(null, m_Sensor))
            {
                m_Sensor.Dispose();
                m_Sensor = null;
            }
        }
    }
}