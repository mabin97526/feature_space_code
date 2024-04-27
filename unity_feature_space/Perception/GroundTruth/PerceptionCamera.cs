using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using UnityEngine.UI;

namespace UnityEngine.Perception.GroundTruth
{
    public class PerceptionCamera : MonoBehaviour
    {
        public bool ManuallyCapture = false;
        public bool ShouldCapture = true;
        public RawImage TestRawImage;
        internal List<ScriptableRenderPass> passes = new List<ScriptableRenderPass>();
    }
}