// Copyright (c) Meta Platforms, Inc. and affiliates.

using Meta.XR.BuildingBlocks;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Events;


// This class loads scene data from either Scene API or a JSON file.
public class RoomLoader : MonoBehaviour
{
    public enum SceneDataSource
    {
        SceneApi,
        StaticMeshData
    }

    // The root transform of the OVRSceneManager.
    [SerializeField] private Transform sceneRoot;

    [SerializeField] private SceneDataLoaderSettings settings;

    [SerializeField] private bool loadAllRooms = false;

    // UnityEvent fired when scene data is loaded.
    public UnityEvent<Transform> SceneDataLoaded = new();

    // UnityEvent fired when scene data is not available.
    public UnityEvent NoSceneModelAvailable = new();

    // UnityEvent fired when a new scene model is available.
    public UnityEvent NewSceneModelAvailable = new();

    public GameObject _meshPrefab;

    private bool _isCompleted = false;

    private void Awake()
    {
        //Assert.IsNotNull(sceneRoot, $"{nameof(sceneRoot)} cannot be null.");
    }


    private void OnDestroy()
    {
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (sceneRoot == null) sceneRoot = transform;
    }
#endif

    private IEnumerator Start()
    {
        var timeout = 10f;
        var startTime = Time.time;
        while (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene))
        {
            if (Time.time - startTime > timeout)
            {
                Debug.LogWarning($"[{nameof(RoomMeshController)}] Spatial Data permission is required to load Room Mesh.");
                yield break;
            }
            yield return null;
        }

       // StartCoroutine(LoadMeshesV2());

        yield return LoadMeshesV2();

        while (!_isCompleted)
        {
            yield return null;
        }
    }


    private IEnumerator LoadMeshesV2()
    {
        yield return null;
        var anchors = new List<OVRAnchor>();
        var result = OVRAnchor.FetchAnchorsAsync(anchors, new OVRAnchor.FetchOptions
        {
            SingleComponentType = typeof(OVRTriangleMesh),
        });

        while (!result.IsCompleted) {
            yield return null;
        }

        // no rooms - call Space Setup or check Scene permission
        if (anchors.Count == 0)
            yield break;

        // get the component to access its data
        foreach (var room in anchors)
        {
            if (!room.TryGetComponent(out OVRTriangleMesh mesh))
                continue;

            if (!room.TryGetComponent(out OVRLocatable locatable))
                continue;

            var localizeTask = locatable.SetEnabledAsync(true);
            while (!localizeTask.IsCompleted) yield return null;

            // use the component helper function to access all child anchors
            InstantiateRoomMesh(room, _meshPrefab);
            yield return null;
        }

        _isCompleted = true;
    }

    private void InstantiateRoomMesh(OVRAnchor anchor, GameObject prefab)
    {
        var _roomMeshAnchor = Instantiate(prefab, Vector3.zero, Quaternion.identity).GetComponent<RoomVisualizer>();
        _roomMeshAnchor.gameObject.name = _meshPrefab.name;
        _roomMeshAnchor.gameObject.SetActive(true);
        _roomMeshAnchor.Initialize(anchor);
    }
}