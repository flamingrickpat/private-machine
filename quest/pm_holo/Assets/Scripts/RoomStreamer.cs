// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Events;
using Classification = OVRSceneManager.Classification;


/// <summary>
/// Post processes scene data after scene loading.
/// </summary>
public class RoomStreamer : MonoBehaviour
{
    // Items in the room probably large enough for phantoms to hop on
    private static readonly string[] WalkableFurniture =
    {
            Classification.Table, Classification.Couch,
            Classification.Other, Classification.Storage,
            Classification.Bed,
        };

    // Items in the room that phantoms shouldn't hop on
    private static readonly string[] RangedTargets =
    {
            Classification.Screen,
            Classification.Lamp,
            Classification.Plant,
        };

    // Items in the room that phantoms can spit goo at
    private static readonly string[] WallMountedTargets =
    {
            Classification.WallArt,
            Classification.WindowFrame,
            Classification.DoorFrame,
        };

    [SerializeField] private UnityEvent<Transform> SceneDataProcessed = new();

    [Tooltip("Remove scene mesh and replace with SceneMesher generated geometry.")]
    [SerializeField]
    private bool forceSceneMesher = false;

    [SerializeField] private bool hideSceneMesh;
    [SerializeField] private bool generateNavMesh = true;

    private readonly List<OVRSemanticClassification> _semanticClassifications =
        new List<OVRSemanticClassification>();

    private bool _sceneReady;

    public string desktopIpAddress = "192.168.1.2";
    public int meshPort = 9000;
    public int transformPort = 9001;

    private UdpClient meshUdpClient;
    private UdpClient transformUdpClient;
    private IPEndPoint meshEndPoint;
    private IPEndPoint transformEndPoint;
    private const int CHUNK_SIZE = 1400; // Safe chunk size for UDP

    public void StreamMeshSync(Mesh mesh, string meshId)
    {
        if (mesh == null) { /* ... error log ... */ return; }
        Debug.Log($"[RoomDataSender-Sync] Preparing to stream mesh '{meshId}' with {mesh.vertexCount} vertices.");

        var meshData = new SerializableMeshData { meshId = meshId };
        foreach (var vert in mesh.vertices) { meshData.vertices.Add(new SerializableVector3(vert)); }
        meshData.triangles.AddRange(mesh.triangles);

        string fullJson = JsonUtility.ToJson(meshData);
        byte[] allBytes = Encoding.UTF8.GetBytes(fullJson);
        int totalChunks = (allBytes.Length + CHUNK_SIZE - 1) / CHUNK_SIZE;

        Debug.Log($"[RoomDataSender-Sync] Chunking '{meshId}' into {totalChunks} chunks.");

        for (int i = 0; i < totalChunks; i++)
        {
            var header = new MeshChunkHeader { meshId = meshId, chunkIndex = i, totalChunks = totalChunks };
            string headerJson = JsonUtility.ToJson(header);
            byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);
            byte[] headerTerminator = Encoding.UTF8.GetBytes("\n\n");

            int offset = i * CHUNK_SIZE;
            int size = Math.Min(CHUNK_SIZE, allBytes.Length - offset);
            byte[] payloadBytes = new byte[size];
            Buffer.BlockCopy(allBytes, offset, payloadBytes, 0, size);

            byte[] packet = new byte[headerBytes.Length + headerTerminator.Length + payloadBytes.Length];
            Buffer.BlockCopy(headerBytes, 0, packet, 0, headerBytes.Length);
            Buffer.BlockCopy(headerTerminator, 0, packet, headerBytes.Length, headerTerminator.Length);
            Buffer.BlockCopy(payloadBytes, 0, packet, headerBytes.Length + headerTerminator.Length, payloadBytes.Length);

            try
            {
                // Use the blocking, synchronous Send method
                meshUdpClient.Send(packet, packet.Length, meshEndPoint);
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomDataSender-Sync] Error sending chunk {i}: {e.Message}");
                return;
            }

            // Use Thread.Sleep instead of Task.Delay
            Thread.Sleep(1);
        }
        Debug.Log($"[RoomDataSender-Sync] Finished streaming mesh '{meshId}'.");
    }


    private void OnEnable()
    {
        meshUdpClient = new UdpClient();
        transformUdpClient = new UdpClient();
        meshEndPoint = new IPEndPoint(IPAddress.Parse(desktopIpAddress), meshPort);
        transformEndPoint = new IPEndPoint(IPAddress.Parse(desktopIpAddress), transformPort);
        Debug.Log($"[RoomDataSender] Networking initialized. Targeting {desktopIpAddress}.");
    }

    private void OnDisable()
    {
    }

    private void OnWorldAligned()
    {
        _sceneReady = true;
    }

    public void PostProcessScene(Transform sceneRoot)
    {
        StartCoroutine(PostProcessInternal(sceneRoot));
    }

    private IEnumerator PostProcessInternal(Transform sceneRoot)
    {
        Bounds GetMeshBounds(Transform meshFilterTransform, List<Vector3> vertices)
        {
            Bounds sceneMeshBounds = default;

            for (var i = 0; i < vertices.Count; i++)
            {
                var worldPos = meshFilterTransform.TransformPoint(vertices[i]);

                if (i == 0)
                {
                    sceneMeshBounds = new Bounds(worldPos, Vector3.zero);
                    continue;
                }

                sceneMeshBounds.Encapsulate(worldPos);
            }

            return sceneMeshBounds;
        }

        // Wait for world alignment to finish.
        do
        {
            yield return null;
        } while (!_sceneReady);

        var rooms = GetComponentsInChildren<OVRSceneRoom>(true);

        Assert.IsTrue(rooms.Length > 0);

        // Process each room, generate navmesh, modify scene mesh etc.
        foreach (var room in rooms)
        {
            while (room.Walls.Length == 0) yield return null;

            Debug.Log($"Post-processing scene: {room.name}");

            List<OVRSemanticClassification> sceneMeshes = new();
            List<OVRSemanticClassification> walkableFurniture = new();
            List<OVRSemanticClassification> targetableFurniture = new();

            Bounds sceneMeshBounds = default;

            room.GetComponentsInChildren(true, _semanticClassifications);

            // All the scene objects we care about should have a semantic classification, regardless of type
            foreach (var semanticObject in _semanticClassifications)
            {
                if (semanticObject.Contains(Classification.GlobalMesh))
                {
                    // To support using static mesh on device.
                    if (semanticObject.TryGetComponent<OVRSceneVolumeMeshFilter>(out var volumeMeshFilter)
                        && volumeMeshFilter.enabled)
                        yield return new WaitUntil(() => volumeMeshFilter.IsCompleted);

                    var meshFilter = semanticObject.GetComponent<MeshFilter>();
                    var vertices = new List<Vector3>();

                    do
                    {
                        yield return null;
                        meshFilter.sharedMesh.GetVertices(vertices);
                    } while (vertices.Count == 0);

                    sceneMeshBounds = GetMeshBounds(meshFilter.transform, vertices);
#if UNITY_EDITOR
                    if (meshFilter == null) meshFilter = semanticObject.GetComponentInChildren<MeshFilter>();

                    if (meshFilter == null)
                        Debug.LogError("No mesh filter on object classified as SceneMesh.", semanticObject);

                    if (semanticObject.TryGetComponent<MeshCollider>(out var meshCollider))
                        while (meshCollider.sharedMesh == null)
                        {
                            Debug.Log("waiting for mesh collider bake!");
                            yield return null;
                        }

                    yield return null;

#endif

                    try
                    {
                        StreamMeshSync(meshFilter.sharedMesh, room.name);
                    }
                    catch (Exception e)
                    {
                        Debug.Log($"[RoomDataSender] Error mesh data: {e.Message}");
                    }


                    Debug.Log($"Scene mesh found with {meshFilter.sharedMesh.triangles.Length} triangles.");
                    sceneMeshes.Add(semanticObject);
                    continue;
                }
            }

            if (forceSceneMesher)
            {
                // Destroy the instantiated scene meshes so we can replace them with SceneMesher objects.
                for (var i = 0; i < sceneMeshes.Count; i++)
                {
                    // FIXME: Disable instead of destroy?
                    Destroy(sceneMeshes[i].gameObject);
                }

                sceneMeshes.Clear();
            }

            if (sceneMeshes.Count == 0)
            {
                // have to wait until the floor's boundary is loaded for meshing to work.
                while (room.Floor.Boundary.Count == 0)
                {
                    yield return null;
                }

                // give the new mesh colliders time to bake
                yield return new WaitForFixedUpdate();
            }
        }
    }
}