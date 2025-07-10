using NUnit.Framework.Internal;
using System.Collections;
using System.Reflection;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;
using static OVRPlugin;

public class RoomVisualizer : MonoBehaviour
{
    public UnityEngine.Mesh _mesh;
    public MeshFilter _meshFilter;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void Awake()
    {
        _mesh = new UnityEngine.Mesh
        {
            name = $"{nameof(RoomMeshAnchor)} (anonymous)"
        };

        if (!TryGetComponent(out _meshFilter))
            _meshFilter = gameObject.AddComponent<MeshFilter>();

        _meshFilter.sharedMesh = _mesh;
    }

    bool IsComponentEnabled<T>(OVRAnchor _anchor) where T : struct, IOVRAnchorComponent<T> => _anchor.TryGetComponent(out T component) && component.IsEnabled;

    private async Task<T> EnableComponent<T>(OVRAnchor _anchor) where T : struct, IOVRAnchorComponent<T>
    {
        if (_anchor.TryGetComponent(out T component))
        {
            await component.SetEnabledAsync(true);
        }

        return component;
    }

    ulong GetHandle(OVRAnchor _anchor)
    {
        ulong handle = (ulong)(typeof(OVRAnchor).GetField("Handle", BindingFlags.Instance).GetValue(_anchor));
        return handle;
    }

    private bool TryUpdateTransform(OVRAnchor _anchor)
    {
        // assume you already have `anchor` from scene query
        if (!_anchor.TryGetComponent(out OVRLocatable locatable))
        {
            return false;
        }

        // one-shot update
        if (locatable.TryGetSceneAnchorPose(out var trackingSpacePose))
        {
            Vector3 worldPos = trackingSpacePose.ComputeWorldPosition(Camera.main) ?? new Vector3(0, 0, 0);
            Quaternion worldOrient = trackingSpacePose.ComputeWorldRotation(Camera.main) ?? Quaternion.identity;
            transform.SetPositionAndRotation(worldPos, worldOrient);
        }
        return true;

        if (!IsComponentEnabled<OVRLocatable>(_anchor))
            return false;

        ulong handle = GetHandle(_anchor);
        var tryLocateSpace = OVRPlugin.TryLocateSpace(handle, OVRPlugin.GetTrackingOriginType(), out var pose,
            out var locationFlags);
        if (!tryLocateSpace || !locationFlags.IsOrientationValid() || !locationFlags.IsPositionValid())
        {
            return false;
        }

        var worldSpacePose = new OVRPose
        {
            position = pose.Position.FromFlippedZVector3f(),
            orientation = pose.Orientation.FromFlippedZQuatf() * Quaternion.Euler(0, 180, 0)
        }.ToWorldSpacePose(Camera.main);
        transform.SetPositionAndRotation(worldSpacePose.position, worldSpacePose.orientation);
        return true;
    }

    internal async void Initialize(OVRAnchor anchor)
    {
        var _anchor = anchor;


        OVRSemanticLabels _labels;
        OVRTriangleMesh _triangleMeshComponent;

        if (TryUpdateTransform(anchor))
        {
            Debug.Log($"[{nameof(RoomMeshAnchor)}][{_anchor.Uuid}] Initial transform set.", gameObject);
        }
        else
        {
            Debug.LogWarning($"[{nameof(RoomMeshAnchor)}][{_anchor.Uuid}] {nameof(OVRPlugin.TryLocateSpace)} failed. The entity may have the wrong initial transform.", gameObject);
        }

        if (!IsComponentEnabled<OVRSemanticLabels>(anchor)) _labels = await EnableComponent<OVRSemanticLabels>(anchor);
        if (!IsComponentEnabled<OVRTriangleMesh>(anchor)) _triangleMeshComponent = await EnableComponent<OVRTriangleMesh>(anchor);

        StartCoroutine(GenerateRoomMesh(_anchor));
    }

    private struct GetTriangleMeshCountsJob : IJob
    {
        public OVRSpace Space;
        [WriteOnly] public NativeArray<int> Results;

        public void Execute()
        {
            Results[0] = -1;
            Results[1] = -1;
            if (OVRPlugin.GetSpaceTriangleMeshCounts(Space, out int vertexCount, out int triangleCount))
            {
                Results[0] = vertexCount;
                Results[1] = triangleCount;
            }
        }
    }

    // IJob wrapper for OVRPlugin.GetSpaceTM
    private struct GetTriangleMeshJob : IJob
    {
        public OVRSpace Space;

        [WriteOnly] public NativeArray<Vector3> Vertices;
        [WriteOnly] public NativeArray<int> Triangles;

        public void Execute() => OVRPlugin.GetSpaceTriangleMesh(Space, Vertices, Triangles);
    }


    // IJob to set vertices/triangles on Unity mesh data, converting from OpenXR
    // to Unity. Ensure that you set mesh data on Mesh after completion.
    private struct PopulateMeshDataJob : IJob
    {
        [ReadOnly] public NativeArray<Vector3> Vertices;
        [ReadOnly] public NativeArray<int> Triangles;

        [WriteOnly]
        public UnityEngine.Mesh.MeshData MeshData;

        public void Execute()
        {
            // assign vertices, converting from OpenXR to Unity
            MeshData.SetVertexBufferParams(Vertices.Length,
                new VertexAttributeDescriptor(VertexAttribute.Position),
                new VertexAttributeDescriptor(VertexAttribute.Normal, stream: 1));
            var vertices = MeshData.GetVertexData<Vector3>();
            for (var i = 0; i < vertices.Length; i++)
            {
                var vertex = Vertices[i];
                vertices[i] = new Vector3(-vertex.x, vertex.y, vertex.z);
            }

            // assign triangles, changing the winding order
            MeshData.SetIndexBufferParams(Triangles.Length, IndexFormat.UInt32);
            var indices = MeshData.GetIndexData<int>();
            for (var i = 0; i < indices.Length; i += 3)
            {
                indices[i + 0] = Triangles[i + 0];
                indices[i + 1] = Triangles[i + 2];
                indices[i + 2] = Triangles[i + 1];
            }

            // lastly, set the sub mesh
            MeshData.subMeshCount = 1;
            MeshData.SetSubMesh(0, new SubMeshDescriptor(0, Triangles.Length));
        }
    }

    // BakeMesh with Physics - this only bakes with default collider options
    // and works on a mesh id. After mesh is baked, it may need assigning
    // to the collider.
    private struct BakeMeshJob : IJob
    {
        public int MeshID;
        public bool Convex;

        public void Execute() => Physics.BakeMesh(MeshID, Convex);
    }

    private static bool IsJobDone(JobHandle job)
    {
        // convenience wrapper to complete job if it's finished
        // use variable to avoid potential race condition
        var completed = job.IsCompleted;
        if (completed) job.Complete();
        return completed;
    }

    private IEnumerator GenerateRoomMesh(OVRAnchor _anchor)
    {

        if (!_anchor.TryGetComponent(out OVRTriangleMesh triMesh))
            yield break;

        // Query counts
        if (triMesh.TryGetCounts(out int vCount, out int tCount))
        {
            using var verts = new NativeArray<Vector3>(vCount, Allocator.TempJob);
            using var tris = new NativeArray<int>(tCount * 3, Allocator.TempJob);

            if (triMesh.TryGetMesh(verts, tris))
            {
                UnityEngine.Mesh unityMesh = new UnityEngine.Mesh { name = $"RoomMesh ({_anchor.Uuid})" };
                unityMesh.SetVertices(verts);
                unityMesh.SetIndices(tris, MeshTopology.Triangles, 0);
                unityMesh.RecalculateNormals();

                _meshFilter.sharedMesh = unityMesh;
            }
        }

        yield break;

        // get mesh data counts
        var vertexCount = -1;
        var triangleCount = -1;
        using (var meshCountResults = new NativeArray<int>(2, Allocator.TempJob))
        {
            ulong handle = GetHandle(_anchor);
            var job = new GetTriangleMeshCountsJob
            {
                Space = handle,
                Results = meshCountResults
            }.Schedule();
            while (!IsJobDone(job))
            {
                yield return null;
            }

            vertexCount = meshCountResults[0];
            triangleCount = meshCountResults[1];
        }

        if (vertexCount == -1)
        {
            yield break;
        }

        // retrieve mesh data, then convert and
        // populate mesh data as dependent job
        var vertices = new NativeArray<Vector3>(vertexCount, Allocator.Persistent);
        var triangles = new NativeArray<int>(triangleCount * 3, Allocator.Persistent);
        var meshDataArray = UnityEngine.Mesh.AllocateWritableMeshData(1);
        var getMeshJob = new GetTriangleMeshJob
        {
            Space = GetHandle(_anchor),
            Vertices = vertices,
            Triangles = triangles
        }.Schedule();
        var populateMeshJob = new PopulateMeshDataJob
        {
            Vertices = vertices,
            Triangles = triangles,
            MeshData = meshDataArray[0]
        }.Schedule(getMeshJob);
        var disposeVerticesJob = JobHandle.CombineDependencies(
            vertices.Dispose(populateMeshJob), triangles.Dispose(populateMeshJob));
        while (!IsJobDone(disposeVerticesJob))
        {
            yield return null;
        }

        // apply data to Unity mesh
        UnityEngine.Mesh.ApplyAndDisposeWritableMeshData(meshDataArray, _mesh);
        _mesh.RecalculateNormals();
        _mesh.RecalculateBounds();

        // bake mesh if we have a collider
        if (TryGetComponent<MeshCollider>(out var collider))
        {
            var job = new BakeMeshJob
            {
                MeshID = _mesh.GetInstanceID(),
                Convex = collider.convex
            }.Schedule();
            while (!IsJobDone(job))
            {
                yield return null;
            }

            collider.sharedMesh = _mesh;
        }
    }


}
