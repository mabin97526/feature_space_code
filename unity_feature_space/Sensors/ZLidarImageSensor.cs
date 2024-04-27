using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Unity.MLAgents.Sensors
{
    public class ZLidarImageRenderTextureSensor: ISensor
    {
        RenderTexture m_RenderTexture;
        private Texture2D[] markerTextures;
        public Camera targetCam;
        public Vector2 markerSize = new Vector2(1, 1);
        public List<Color> marketColors;
        public int startLayer;
        public RawImage rawImage;
        public float random;
        Texture2D original;
        List<int> layers;
        Texture2D newTexture;
        RayPerceptionSensorComponent3D m_rayPerception;
        RenderTextureSensor m_RenderTextureSensor;
        public ZLidarImageRenderTextureSensor(
            int width, int height,
            RayPerceptionSensorComponent3D raysensor,
            Camera camera,
            string name,
            SensorCompressionType compressionType,
            List<Color> marketColors,
            int startLayer,
            Vector2 markerSize,
            RawImage rawImage,
            float m_random
            )
        {
            m_RenderTexture = new RenderTexture(width, height, 8, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            m_rayPerception = raysensor;
            targetCam = camera;
            m_RenderTextureSensor = new RenderTextureSensor(m_RenderTexture, false, name, compressionType);
            this.marketColors = marketColors;
            this.startLayer = startLayer;
            this.markerSize = markerSize;
            this.rawImage = rawImage;
            this.random = m_random;
            markerTextures = new Texture2D[marketColors.Count];
            for (int i = 0; i < this.marketColors.Count; i++)
            {
                markerTextures[i] = new Texture2D(1, 1);
                markerTextures[i].SetPixel(0, 0, marketColors[i]);
                markerTextures[i].Apply();
            }
        }

        public ObservationSpec GetObservationSpec()
        {
            return m_RenderTextureSensor.GetObservationSpec();
        }

        public int Write(ObservationWriter writer)
        {
            return m_RenderTextureSensor.Write(writer);
        }

        public byte[] GetCompressedObservation()
        {
            return m_RenderTextureSensor.GetCompressedObservation();
        }

        public void Update()
        {
            if(newTexture!=null)
            {
                Object.Destroy(newTexture);
            }
            
            targetCam.Render();
            original = GetTexture2D(targetCam.targetTexture);
            newTexture = new Texture2D(original.width, original.height);
            newTexture.SetPixels32(original.GetPixels32());
            newTexture.Apply();
            List<Vector2> points = GetCoord(targetCam, m_rayPerception);
            if (points == null)
                return ;
            for (int i = 0; i < points.Count; i++)
            {
                DrawPoint(newTexture, points[i], markerSize, markerTextures[layers[i]]);
            }
            //可视化
            rawImage.texture = newTexture;
            RenderTexture tmpRT = RenderTexture.GetTemporary(m_RenderTexture.width, m_RenderTexture.height);
            RenderTexture.active = tmpRT;
            Graphics.Blit(newTexture, tmpRT);
            Graphics.Blit(tmpRT, m_RenderTexture);
            RenderTexture.ReleaseTemporary(tmpRT);

            Object.Destroy(original);
            layers = null ;
            points = null ;
        }
        

        public void Reset()
        {
            
        }

        public CompressionSpec GetCompressionSpec()
        {
            return m_RenderTextureSensor.GetCompressionSpec();
        }

        public string GetName()
        {
            return m_RenderTextureSensor.GetName();
        }
        Texture2D GetTexture2D(Texture texture)
        {
            if (texture is Texture2D)
            {
                return (Texture2D)texture;
            }
            else if (texture is RenderTexture)
            {
                RenderTexture renderTexture = (RenderTexture)texture;
                Texture2D result = new Texture2D(renderTexture.width, renderTexture.height);
                RenderTexture.active = renderTexture;
                result.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
                result.Apply();
                RenderTexture.active = null;
                return result;
            }
            else
            {
                return null;
            }
        }
        /// <summary>
        ///   获取命中雷达点在目标相机的像素坐标，并保存其命中物体的layer
        /// </summary>
        /// <param name="targetCamera"></param>
        /// <param name="rayPerception"></param>
        /// <returns></returns>
        public List<Vector2> GetCoord(Camera targetCamera, RayPerceptionSensorComponent3D rayPerception)
        {
            List<Vector2> viewPoints = new List<Vector2>();
            if(layers == null)
            {
                layers = new List<int>();
            }
            //layers = new List<int>();
            var outputs = rayPerception?.RaySensor?.RayPerceptionOutput?.RayOutputs;
            if (outputs == null)
                return null;
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i].HasHit == true )
                {
                    if(random > 0)
                    {
                        if(Random.Range(0f, 1f) >= random)
                        {
                            Vector3 point = targetCam.WorldToScreenPoint(outputs[i].EndPositionWorld);
                            //UnityEngine.Debug.Log(point);
                            viewPoints.Add(new Vector2(point.x, point.y));
                            layers.Add(outputs[i].HitGameObject.layer - startLayer);
                        }
                        
                    }
                    else
                    {
                        Vector3 point = targetCam.WorldToScreenPoint(outputs[i].EndPositionWorld);
                        //UnityEngine.Debug.Log(point);
                        viewPoints.Add(new Vector2(point.x, point.y));
                        layers.Add(outputs[i].HitGameObject.layer - startLayer);
                    }
                     
                    
                }
            }
            return viewPoints;
        }
        /// <summary>
        ///  绘制目标点
        /// </summary>
        /// <param name="texture"></param>
        /// <param name="coordinates"></param>
        /// <param name="size"></param>
        /// <param name="markerTexture"></param>
        private void DrawPoint(Texture2D texture, Vector2 coordinates, Vector2 size, Texture2D markerTexture)
        {
            Color[] markerPixels = markerTexture.GetPixels();
            int markerWidth = markerTexture.width;
            int markerHeight = markerTexture.height;

            for (int y = 0; y < size.y; y++)
            {
                for (int x = 0; x < size.x; x++)
                {
                    int targetX = Mathf.FloorToInt(coordinates.x) + x;
                    int targetY = Mathf.FloorToInt(coordinates.y) + y;

                    if (targetX >= 0 && targetX < texture.width && targetY >= 0 && targetY < texture.height)
                    {
                        // 根据缩放计算标记纹理中的像素索引
                        int markerX = Mathf.FloorToInt((x / size.x) * markerWidth);
                        int markerY = Mathf.FloorToInt((y / size.y) * markerHeight);

                        texture.SetPixel(targetX, targetY, markerPixels[markerY * markerWidth + markerX]);
                    }
                }
            }

            texture.Apply();
        }
    }

    public class ZLidarImageSensor : SensorComponent
    {
        [Tooltip("需要融合的具体相机，必须配置")]
        public Camera camera;

        public string SensorName = "LidarImageSensor";
        public SensorCompressionType Compression = SensorCompressionType.PNG;
        public int width, height;
        [Tooltip("绘制点云的大小")]
        public Vector2 markerSize = new Vector2(1, 1);
        [Tooltip("雷达检测到的各类物体的具体颜色")]
        public List<Color> marketColors;
        [Tooltip("起始的物体layer号")]
        public int startLayer;
        [Tooltip("用于展示效果")]
        public RawImage rawImage;
        [Tooltip("雷达")]
        public RayPerceptionSensorComponent3D m_rayPerception;

        ZLidarImageRenderTextureSensor m_Sensor;
        [Tooltip("随机化参数")]
        public float m_random;
        public override ISensor[] CreateSensors()
        {
            if(m_Sensor!=null)
            {
                return new ISensor[] { m_Sensor };
            }
            m_Sensor = new ZLidarImageRenderTextureSensor(width, height,
                m_rayPerception,
                camera,
                SensorName,
                Compression,
                marketColors,
                startLayer,
                markerSize,
                rawImage,
                m_random
                );
            return new ISensor[] { m_Sensor };
        }


         
    }
}


