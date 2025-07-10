using UnityEngine;
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Linq;

public class RoomDataReceiver : MonoBehaviour
{
    private const int MESH_PORT = 9000;
    private const int TRANSFORM_PORT = 9001;

    public GameObject playerHeadPrefab;
    private GameObject playerHeadInstance;

    // --- Data Structures for Reassembly ---
    // Key: meshId, Value: A structure holding the reassembly data
    private Dictionary<string, MeshReassemblyData> meshBuffer = new Dictionary<string, MeshReassemblyData>();
    private class MeshReassemblyData
    {
        public int totalChunks;
        public Dictionary<int, byte[]> receivedChunks = new Dictionary<int, byte[]>();
        public DateTime lastReceivedTime;
    }

    // --- Main Thread Data Queue ---
    // We queue the completed data to be processed safely in Update()
    private readonly Queue<SerializableMeshData> completedMeshDataQueue = new Queue<SerializableMeshData>();
    private volatile SerializableTransformData receivedTransformData;

    private Thread meshListenerThread;
    private Thread transformListenerThread;
    private UdpClient meshUdpClient;
    private UdpClient transformUdpClient;

    void Start()
    {
        if (playerHeadPrefab != null)
        {
            playerHeadInstance = Instantiate(playerHeadPrefab, Vector3.zero, Quaternion.identity);
            playerHeadInstance.SetActive(false);
        }

        meshListenerThread = new Thread(MeshListener);
        meshListenerThread.IsBackground = true;
        meshListenerThread.Start();

        transformListenerThread = new Thread(TransformListener);
        transformListenerThread.IsBackground = true;
        transformListenerThread.Start();
    }

    void Update()
    {
        // Process completed meshes on the main thread
        lock (completedMeshDataQueue)
        {
            while (completedMeshDataQueue.Count > 0)
            {
                var meshData = completedMeshDataQueue.Dequeue();
                CreateOrUpdateRoomMesh(meshData);
            }
        }

        // Update player transform
        if (receivedTransformData != null && playerHeadInstance != null)
        {
            if (!playerHeadInstance.activeInHierarchy) playerHeadInstance.SetActive(true);
            playerHeadInstance.transform.SetPositionAndRotation(
                receivedTransformData.position.ToVector3(),
                receivedTransformData.rotation.ToQuaternion()
            );
        }
    }

    private void MeshListener()
    {
        meshUdpClient = new UdpClient(MESH_PORT);
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

        while (true)
        {
            byte[] packet = meshUdpClient.Receive(ref anyIP);

            // 1. Separate header from payload
            string packetString = Encoding.UTF8.GetString(packet);
            string separator = "\n\n";
            int headerEndIndex = packetString.IndexOf(separator);
            if (headerEndIndex == -1) continue;

            string headerJson = packetString.Substring(0, headerEndIndex);
            var header = JsonUtility.FromJson<MeshChunkHeader>(headerJson);

            int payloadStartIndex = Encoding.UTF8.GetBytes(headerJson).Length + Encoding.UTF8.GetBytes(separator).Length;
            byte[] payload = new byte[packet.Length - payloadStartIndex];
            Buffer.BlockCopy(packet, payloadStartIndex, payload, 0, payload.Length);

            // 2. Store the chunk
            lock (meshBuffer)
            {
                if (!meshBuffer.ContainsKey(header.meshId))
                {
                    meshBuffer[header.meshId] = new MeshReassemblyData { totalChunks = header.totalChunks };
                }
                var reassemblyData = meshBuffer[header.meshId];
                reassemblyData.receivedChunks[header.chunkIndex] = payload;
                reassemblyData.lastReceivedTime = DateTime.UtcNow;

                // 3. Check if assembly is complete
                if (reassemblyData.receivedChunks.Count == reassemblyData.totalChunks)
                {
                    Debug.Log($"[RoomDataReceiver] All {reassemblyData.totalChunks} chunks received for '{header.meshId}'. Reassembling...");

                    // Reassemble the full byte array
                    var orderedChunks = reassemblyData.receivedChunks.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value);
                    byte[] fullBytes = orderedChunks.SelectMany(a => a).ToArray();
                    string fullJson = Encoding.UTF8.GetString(fullBytes);

                    var completedMeshData = JsonUtility.FromJson<SerializableMeshData>(fullJson);

                    // Add to main thread queue and clean up
                    lock (completedMeshDataQueue)
                    {
                        completedMeshDataQueue.Enqueue(completedMeshData);
                    }
                    meshBuffer.Remove(header.meshId);
                }
            }
        }
    }

    private void TransformListener()
    {
        transformUdpClient = new UdpClient(TRANSFORM_PORT);
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            byte[] bytes = transformUdpClient.Receive(ref anyIP);
            string json = Encoding.UTF8.GetString(bytes);
            receivedTransformData = JsonUtility.FromJson<SerializableTransformData>(json);
        }
    }

    private void CreateOrUpdateRoomMesh(SerializableMeshData meshData)
    {
        GameObject existingRoom = GameObject.Find(meshData.meshId);
        if (existingRoom == null)
        {
            existingRoom = new GameObject(meshData.meshId);
            existingRoom.AddComponent<MeshFilter>();
            existingRoom.AddComponent<MeshRenderer>().material = new Material(Shader.Find("Standard"));
            existingRoom.AddComponent<MeshCollider>();
        }

        Mesh mesh = new Mesh { name = meshData.meshId };
        var vertices = new Vector3[meshData.vertices.Count];
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] = meshData.vertices[i].ToVector3();
        }

        mesh.vertices = vertices;
        mesh.triangles = meshData.triangles.ToArray();
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();

        existingRoom.GetComponent<MeshFilter>().mesh = mesh;
        existingRoom.GetComponent<MeshCollider>().sharedMesh = mesh;

        Debug.Log($"[RoomDataReceiver] Successfully created/updated mesh for '{meshData.meshId}'.");
    }

    void OnApplicationQuit()
    {
        meshListenerThread?.Abort();
        transformListenerThread?.Abort();
        meshUdpClient?.Close();
        transformUdpClient?.Close();
    }
}