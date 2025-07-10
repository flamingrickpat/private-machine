using System;
using System.Collections.Generic;
using UnityEngine;

// A version of Vector3 that can be serialized to JSON.
[Serializable]
public class SerializableVector3
{
    public float x, y, z;
    public SerializableVector3(Vector3 v) { x = v.x; y = v.y; z = v.z; }
    public Vector3 ToVector3() { return new Vector3(x, y, z); }
}

// A version of Quaternion that can be serialized to JSON.
[Serializable]
public class SerializableQuaternion
{
    public float x, y, z, w;
    public SerializableQuaternion(Quaternion q) { x = q.x; y = q.y; z = q.z; w = q.w; }
    public Quaternion ToQuaternion() { return new Quaternion(x, y, z, w); }
}

// The complete data for a single mesh. This is what we CHUNK.
[Serializable]
public class SerializableMeshData
{
    public string meshId; // e.g., the name of the room
    public List<SerializableVector3> vertices = new List<SerializableVector3>();
    public List<int> triangles = new List<int>();
}

// The data for the player's head transform. This is sent in one packet.
[Serializable]
public class SerializableTransformData
{
    public SerializableVector3 position;
    public SerializableQuaternion rotation;
}

// This is the HEADER for our mesh chunk packets.
// We will serialize this to JSON and attach it to the front of each chunk.
[Serializable]
public class MeshChunkHeader
{
    public string meshId;
    public int chunkIndex;
    public int totalChunks;
}
